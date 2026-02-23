//! HNSW index implementation with HDF5 serialization.

use std::collections::{BinaryHeap, HashSet};

use rustyhdf5_format::error::FormatError;
use rustyhdf5_format::file_writer::{AttrValue, FileWriter as FmtWriter};
use rustyhdf5_format::signature::find_signature;
use rustyhdf5_format::superblock::Superblock;
use rustyhdf5_format::group_v2::resolve_path_any;
use rustyhdf5_format::object_header::ObjectHeader;
use rustyhdf5_format::message_type::MessageType;
use rustyhdf5_format::data_layout::DataLayout;
use rustyhdf5_format::dataspace::Dataspace;
use rustyhdf5_format::datatype::Datatype;
use rustyhdf5_format::data_read::{read_as_f32, read_as_i32, read_raw_data_full};
use rustyhdf5_format::filter_pipeline::FilterPipeline;
use rustyhdf5_format::attribute::extract_attributes_full;
use rustyhdf5_io::FileWriter as IoFileWriter;
use rustyhdf5_io::HDF5ReadWrite;

/// Distance metric for the HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance.
    L2,
    /// Cosine distance (1 - cosine_similarity).
    Cosine,
}

impl DistanceMetric {
    fn as_str(self) -> &'static str {
        match self {
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "l2" => Some(DistanceMetric::L2),
            "cosine" => Some(DistanceMetric::Cosine),
            _ => None,
        }
    }
}

/// Compute distance between two vectors using the given metric.
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => {
            let mut sum = 0.0f32;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                sum += d * d;
            }
            sum.sqrt()
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for i in 0..a.len() {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom < f32::EPSILON {
                1.0
            } else {
                1.0 - (dot / denom)
            }
        }
    }
}

/// Assign a random level to a new node based on the HNSW probability distribution.
///
/// Uses a deterministic approach based on the node index for reproducibility.
fn assign_level(node_id: usize, m: usize) -> usize {
    let ml = 1.0 / (m as f64).ln();
    // Use a simple hash-based pseudo-random for reproducibility
    let hash = splitmix64(node_id as u64);
    let uniform = (hash >> 11) as f64 / (1u64 << 53) as f64;
    (-uniform.ln() * ml).floor() as usize
}

/// Simple splitmix64 hash for deterministic level assignment.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Candidate neighbor for priority queue operations.
#[derive(Debug, Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Max-heap candidate (furthest first).
#[derive(Debug, Clone)]
struct FarCandidate {
    id: usize,
    distance: f32,
}

impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.id == other.id
    }
}

impl Eq for FarCandidate {}

impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
///
/// Supports building an index from vectors, searching for nearest neighbors,
/// and serializing/deserializing to HDF5 format.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    /// All vectors in the index.
    vectors: Vec<Vec<f32>>,
    /// Adjacency lists per layer. `graph[layer][node]` = list of neighbor IDs.
    graph: Vec<Vec<Vec<usize>>>,
    /// Entry point node ID.
    entry_point: usize,
    /// Maximum number of connections per node (per layer).
    m: usize,
    /// Maximum connections for layer 0 (typically 2*m).
    m_max0: usize,
    /// ef parameter used during construction.
    ef_construction: usize,
    /// Maximum layer assigned to each node.
    node_levels: Vec<usize>,
    /// Distance metric.
    metric: DistanceMetric,
}

impl HnswIndex {
    /// Build an HNSW index from a set of vectors.
    ///
    /// # Parameters
    /// - `vectors`: The vectors to index. All must have the same dimension.
    /// - `m`: Maximum number of connections per node (higher = more accurate, more memory).
    /// - `ef_construction`: Size of the dynamic candidate list during construction.
    ///
    /// Uses L2 distance by default. Use [`build_with_metric`] to specify the metric.
    pub fn build(vectors: &[Vec<f32>], m: usize, ef_construction: usize) -> Self {
        Self::build_with_metric(vectors, m, ef_construction, DistanceMetric::L2)
    }

    /// Build an HNSW index with a specific distance metric.
    pub fn build_with_metric(
        vectors: &[Vec<f32>],
        m: usize,
        ef_construction: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert!(!vectors.is_empty(), "cannot build index from empty vectors");
        assert!(m >= 2, "m must be at least 2");
        let dim = vectors[0].len();
        for v in vectors {
            assert_eq!(v.len(), dim, "all vectors must have the same dimension");
        }

        let m_max0 = m * 2;
        let n = vectors.len();

        // Assign levels to all nodes
        let mut node_levels = Vec::with_capacity(n);
        let mut max_level = 0;
        for i in 0..n {
            let level = assign_level(i, m);
            if level > max_level {
                max_level = level;
            }
            node_levels.push(level);
        }

        // Initialize graph layers
        let num_layers = max_level + 1;
        let mut graph: Vec<Vec<Vec<usize>>> = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            graph.push(vec![Vec::new(); n]);
        }

        let mut entry_point = 0;
        let mut ep_level = node_levels[0];

        // Insert nodes one by one
        for i in 1..n {
            let node_level = node_levels[i];
            let mut ep = entry_point;

            // Phase 1: greedy search from top layer down to node_level + 1
            let start_layer = ep_level;
            for layer in (node_level + 1..=start_layer).rev() {
                ep = greedy_closest(vectors, &graph[layer], &vectors[i], ep, metric);
            }

            // Phase 2: search and connect at layers node_level down to 0
            let bottom = if node_level < start_layer {
                node_level
            } else {
                start_layer
            };
            for layer in (0..=bottom).rev() {
                let max_conn = if layer == 0 { m_max0 } else { m };

                let neighbors = search_layer(
                    vectors,
                    &graph[layer],
                    &vectors[i],
                    ep,
                    ef_construction,
                    metric,
                );

                // Select up to m closest neighbors
                let selected: Vec<usize> = neighbors
                    .iter()
                    .take(max_conn)
                    .map(|c| c.id)
                    .collect();

                // Add bidirectional connections
                graph[layer][i] = selected.clone();
                for &neighbor in &selected {
                    graph[layer][neighbor].push(i);
                    // Prune if over limit
                    if graph[layer][neighbor].len() > max_conn {
                        prune_connections(
                            vectors,
                            &mut graph[layer][neighbor],
                            neighbor,
                            max_conn,
                            metric,
                        );
                    }
                }

                if !selected.is_empty() {
                    ep = selected[0];
                }
            }

            // Update entry point if this node has a higher level
            if node_level > ep_level {
                entry_point = i;
                ep_level = node_level;
            }
        }

        Self {
            vectors: vectors.to_vec(),
            graph,
            entry_point,
            m,
            m_max0,
            ef_construction,
            node_levels,
            metric,
        }
    }

    /// Search the index for the `k` nearest neighbors to the query vector.
    ///
    /// # Parameters
    /// - `query`: The query vector.
    /// - `k`: Number of nearest neighbors to return.
    /// - `ef`: Size of the dynamic candidate list during search (must be >= k).
    ///
    /// # Returns
    /// A vector of `(id, distance)` pairs sorted by distance (closest first).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            query.len(),
            self.vectors[0].len(),
            "query dimension mismatch"
        );
        let ef = ef.max(k);

        let mut ep = self.entry_point;
        let top_layer = self.graph.len().saturating_sub(1);

        // Greedy search from top layer down to layer 1
        for layer in (1..=top_layer).rev() {
            ep = greedy_closest(&self.vectors, &self.graph[layer], query, ep, self.metric);
        }

        // Search layer 0 with ef candidates
        let candidates = search_layer(
            &self.vectors,
            &self.graph[0],
            query,
            ep,
            ef,
            self.metric,
        );

        candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect()
    }

    /// Save the index to an HDF5 file via the given writer.
    pub fn save_to_hdf5(&self, writer: &mut IoFileWriter) -> Result<(), FormatError> {
        let bytes = self.to_hdf5_bytes()?;
        writer
            .write_all_bytes(&bytes)
            .map_err(|e| FormatError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Serialize the index to HDF5 bytes.
    pub fn to_hdf5_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let mut fw = FmtWriter::new();
        let n = self.vectors.len();
        let dim = if n > 0 { self.vectors[0].len() } else { 0 };

        // Flatten vectors into a 1D array for storage
        let flat_vectors: Vec<f32> = self.vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let mut group = fw.create_group("ann");

        group
            .create_dataset("vectors")
            .with_f32_data(&flat_vectors)
            .with_shape(&[n as u64, dim as u64])
            .set_attr("rows", AttrValue::I64(n as i64))
            .set_attr("cols", AttrValue::I64(dim as i64));

        // Serialize graph layers: store as flat i32 arrays with metadata
        let num_layers = self.graph.len();
        for (layer_idx, layer) in self.graph.iter().enumerate() {
            // Flatten: for each node, store [count, neighbor1, neighbor2, ...]
            let mut flat: Vec<i32> = Vec::new();
            for neighbors in layer {
                flat.push(neighbors.len() as i32);
                for &n_id in neighbors {
                    flat.push(n_id as i32);
                }
            }

            let ds_name = format!("graph_layer_{layer_idx}");
            group
                .create_dataset(&ds_name)
                .with_i32_data(&flat)
                .set_attr("layer", AttrValue::I64(layer_idx as i64));
        }

        // Store config as attributes on a small dataset
        let node_levels_i32: Vec<i32> = self.node_levels.iter().map(|&l| l as i32).collect();
        group
            .create_dataset("config")
            .with_i32_data(&node_levels_i32)
            .set_attr("m", AttrValue::I64(self.m as i64))
            .set_attr("ef_construction", AttrValue::I64(self.ef_construction as i64))
            .set_attr("entry_point", AttrValue::I64(self.entry_point as i64))
            .set_attr("num_layers", AttrValue::I64(num_layers as i64))
            .set_attr(
                "metric",
                AttrValue::String(self.metric.as_str().to_string()),
            )
            .set_attr("num_vectors", AttrValue::I64(n as i64))
            .set_attr("dimension", AttrValue::I64(dim as i64));

        let finished = group.finish();
        fw.add_group(finished);
        fw.finish()
    }

    /// Load an HNSW index from HDF5 bytes.
    ///
    /// The HDF5 data must contain the `/ann/vectors`, `/ann/graph_layer_*`,
    /// and `/ann/config` datasets as produced by [`to_hdf5_bytes`].
    pub fn load_from_hdf5(data: &[u8], _path: &str) -> Result<Self, FormatError> {
        let sig_offset = find_signature(data)?;
        let sb = Superblock::parse(data, sig_offset)?;

        // Read config dataset and its attributes
        let config_attrs = read_dataset_attrs(data, &sb, "ann/config")?;
        let config_raw = read_dataset_raw(data, &sb, "ann/config")?;
        let config_dt = read_dataset_datatype(data, &sb, "ann/config")?;
        let node_levels_i32 = read_as_i32(&config_raw, &config_dt)?;

        let m = get_attr_i64(&config_attrs, "m")? as usize;
        let ef_construction = get_attr_i64(&config_attrs, "ef_construction")? as usize;
        let entry_point = get_attr_i64(&config_attrs, "entry_point")? as usize;
        let num_layers = get_attr_i64(&config_attrs, "num_layers")? as usize;
        let n = get_attr_i64(&config_attrs, "num_vectors")? as usize;
        let dim = get_attr_i64(&config_attrs, "dimension")? as usize;
        let metric_str = get_attr_string(&config_attrs, "metric")?;
        let metric = DistanceMetric::from_str(&metric_str)
            .ok_or_else(|| FormatError::SerializationError(format!("unknown metric: {metric_str}")))?;

        let node_levels: Vec<usize> = node_levels_i32.iter().map(|&l| l as usize).collect();

        // Read vectors
        let vectors_raw = read_dataset_raw(data, &sb, "ann/vectors")?;
        let vectors_dt = read_dataset_datatype(data, &sb, "ann/vectors")?;
        let flat_vectors = read_as_f32(&vectors_raw, &vectors_dt)?;
        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            if end > flat_vectors.len() {
                return Err(FormatError::DataSizeMismatch {
                    expected: end,
                    actual: flat_vectors.len(),
                });
            }
            vectors.push(flat_vectors[start..end].to_vec());
        }

        // Read graph layers
        let mut graph = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let ds_name = format!("ann/graph_layer_{layer_idx}");
            let layer_raw = read_dataset_raw(data, &sb, &ds_name)?;
            let layer_dt = read_dataset_datatype(data, &sb, &ds_name)?;
            let flat = read_as_i32(&layer_raw, &layer_dt)?;

            let mut layer_graph = Vec::with_capacity(n);
            let mut pos = 0;
            while pos < flat.len() {
                let count = flat[pos] as usize;
                pos += 1;
                let mut neighbors = Vec::with_capacity(count);
                for _ in 0..count {
                    if pos >= flat.len() {
                        return Err(FormatError::SerializationError(
                            "truncated graph data".into(),
                        ));
                    }
                    neighbors.push(flat[pos] as usize);
                    pos += 1;
                }
                layer_graph.push(neighbors);
            }
            // Pad with empty if needed (nodes not present at this layer)
            while layer_graph.len() < n {
                layer_graph.push(Vec::new());
            }
            graph.push(layer_graph);
        }

        Ok(Self {
            vectors,
            graph,
            entry_point,
            m,
            m_max0: m * 2,
            ef_construction,
            node_levels,
            metric,
        })
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Returns the dimension of vectors in the index.
    pub fn dimension(&self) -> usize {
        if self.vectors.is_empty() {
            0
        } else {
            self.vectors[0].len()
        }
    }

    /// Returns the number of layers in the graph.
    pub fn num_layers(&self) -> usize {
        self.graph.len()
    }

    /// Returns the distance metric used by this index.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Returns the maximum number of connections at layer 0.
    pub fn m_max0(&self) -> usize {
        self.m_max0
    }
}

// ---------------------------------------------------------------------------
// Internal HNSW algorithms
// ---------------------------------------------------------------------------

/// Greedy search: find the single closest node to `query` starting from `ep`.
fn greedy_closest(
    vectors: &[Vec<f32>],
    layer: &[Vec<usize>],
    query: &[f32],
    mut ep: usize,
    metric: DistanceMetric,
) -> usize {
    let mut best_dist = compute_distance(query, &vectors[ep], metric);
    loop {
        let mut changed = false;
        for &neighbor in &layer[ep] {
            let d = compute_distance(query, &vectors[neighbor], metric);
            if d < best_dist {
                best_dist = d;
                ep = neighbor;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    ep
}

/// Search a single layer for the ef closest nodes to `query`.
fn search_layer(
    vectors: &[Vec<f32>],
    layer: &[Vec<usize>],
    query: &[f32],
    ep: usize,
    ef: usize,
    metric: DistanceMetric,
) -> Vec<Candidate> {
    let ep_dist = compute_distance(query, &vectors[ep], metric);

    // Min-heap of candidates to explore
    let mut candidates = BinaryHeap::new();
    candidates.push(Candidate {
        id: ep,
        distance: ep_dist,
    });

    // Max-heap of current results (furthest first)
    let mut results = BinaryHeap::new();
    results.push(FarCandidate {
        id: ep,
        distance: ep_dist,
    });

    let mut visited = HashSet::new();
    visited.insert(ep);

    while let Some(closest) = candidates.pop() {
        let furthest_dist = results.peek().map_or(f32::MAX, |f| f.distance);
        if closest.distance > furthest_dist && results.len() >= ef {
            break;
        }

        for &neighbor in &layer[closest.id] {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            let d = compute_distance(query, &vectors[neighbor], metric);
            let furthest_dist = results.peek().map_or(f32::MAX, |f| f.distance);

            if d < furthest_dist || results.len() < ef {
                candidates.push(Candidate {
                    id: neighbor,
                    distance: d,
                });
                results.push(FarCandidate {
                    id: neighbor,
                    distance: d,
                });
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    // Convert to sorted vec
    let mut result: Vec<Candidate> = results
        .into_iter()
        .map(|f| Candidate {
            id: f.id,
            distance: f.distance,
        })
        .collect();
    result.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

/// Prune connections for a node to keep only the closest `max_conn` neighbors.
fn prune_connections(
    vectors: &[Vec<f32>],
    neighbors: &mut Vec<usize>,
    node: usize,
    max_conn: usize,
    metric: DistanceMetric,
) {
    if neighbors.len() <= max_conn {
        return;
    }
    let mut scored: Vec<(usize, f32)> = neighbors
        .iter()
        .map(|&n| (n, compute_distance(&vectors[node], &vectors[n], metric)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(max_conn);
    *neighbors = scored.into_iter().map(|(id, _)| id).collect();
}

// ---------------------------------------------------------------------------
// HDF5 reading helpers
// ---------------------------------------------------------------------------

fn read_dataset_raw(data: &[u8], sb: &Superblock, path: &str) -> Result<Vec<u8>, FormatError> {
    let addr = resolve_path_any(data, sb, path)?;
    let header = ObjectHeader::parse(data, addr as usize, sb.offset_size, sb.length_size)?;

    let dt_msg = header.messages.iter().find(|m| m.msg_type == MessageType::Datatype)
        .ok_or(FormatError::DatasetMissingData)?;
    let (datatype, _) = Datatype::parse(&dt_msg.data)?;

    let ds_msg = header.messages.iter().find(|m| m.msg_type == MessageType::Dataspace)
        .ok_or(FormatError::DatasetMissingShape)?;
    let dataspace = Dataspace::parse(&ds_msg.data, sb.length_size)?;

    let dl_msg = header.messages.iter().find(|m| m.msg_type == MessageType::DataLayout)
        .ok_or(FormatError::DatasetMissingData)?;
    let layout = DataLayout::parse(&dl_msg.data, sb.offset_size, sb.length_size)?;

    let pipeline = header.messages.iter()
        .find(|m| m.msg_type == MessageType::FilterPipeline)
        .and_then(|msg| FilterPipeline::parse(&msg.data).ok());

    read_raw_data_full(data, &layout, &dataspace, &datatype, pipeline.as_ref(), sb.offset_size, sb.length_size)
}

fn read_dataset_datatype(data: &[u8], sb: &Superblock, path: &str) -> Result<Datatype, FormatError> {
    let addr = resolve_path_any(data, sb, path)?;
    let header = ObjectHeader::parse(data, addr as usize, sb.offset_size, sb.length_size)?;
    let dt_msg = header.messages.iter().find(|m| m.msg_type == MessageType::Datatype)
        .ok_or(FormatError::DatasetMissingData)?;
    let (datatype, _) = Datatype::parse(&dt_msg.data)?;
    Ok(datatype)
}

fn read_dataset_attrs(
    data: &[u8],
    sb: &Superblock,
    path: &str,
) -> Result<Vec<(String, AttrValue)>, FormatError> {
    let addr = resolve_path_any(data, sb, path)?;
    let header = ObjectHeader::parse(data, addr as usize, sb.offset_size, sb.length_size)?;
    let attr_msgs = extract_attributes_full(data, &header, sb.offset_size, sb.length_size)?;

    let mut result = Vec::new();
    for attr in &attr_msgs {
        let name = attr.name.clone();
        if let Some(val) = decode_simple_attr(attr) {
            result.push((name, val));
        }
    }
    Ok(result)
}

fn decode_simple_attr(attr: &rustyhdf5_format::attribute::AttributeMessage) -> Option<AttrValue> {
    let raw = &attr.raw_data;
    match &attr.datatype {
        Datatype::FixedPoint { size, signed, .. } => {
            if *signed {
                match size {
                    8 => {
                        if raw.len() >= 8 {
                            let val = i64::from_le_bytes(raw[..8].try_into().ok()?);
                            Some(AttrValue::I64(val))
                        } else {
                            None
                        }
                    }
                    4 => {
                        if raw.len() >= 4 {
                            let val = i32::from_le_bytes(raw[..4].try_into().ok()?) as i64;
                            Some(AttrValue::I64(val))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            } else if raw.len() >= *size as usize {
                let val = u64::from_le_bytes({
                    let mut buf = [0u8; 8];
                    buf[..*size as usize].copy_from_slice(&raw[..*size as usize]);
                    buf
                });
                Some(AttrValue::U64(val))
            } else {
                None
            }
        }
        Datatype::FloatingPoint { size: 8, .. } => {
            if raw.len() >= 8 {
                let val = f64::from_le_bytes(raw[..8].try_into().ok()?);
                Some(AttrValue::F64(val))
            } else {
                None
            }
        }
        Datatype::String { size, .. } => {
            let s = std::str::from_utf8(&raw[..*size as usize]).ok()?;
            Some(AttrValue::String(s.trim_end_matches('\0').to_string()))
        }
        _ => None,
    }
}

fn get_attr_i64(attrs: &[(String, AttrValue)], name: &str) -> Result<i64, FormatError> {
    for (n, v) in attrs {
        if n == name {
            return match v {
                AttrValue::I64(val) => Ok(*val),
                AttrValue::U64(val) => Ok(*val as i64),
                _ => Err(FormatError::SerializationError(format!(
                    "attribute {name} is not an integer"
                ))),
            };
        }
    }
    Err(FormatError::SerializationError(format!(
        "missing attribute: {name}"
    )))
}

fn get_attr_string(attrs: &[(String, AttrValue)], name: &str) -> Result<String, FormatError> {
    for (n, v) in attrs {
        if n == name {
            return match v {
                AttrValue::String(s) => Ok(s.clone()),
                _ => Err(FormatError::SerializationError(format!(
                    "attribute {name} is not a string"
                ))),
            };
        }
    }
    Err(FormatError::SerializationError(format!(
        "missing attribute: {name}"
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut vectors = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                state = splitmix64(state);
                let val = (state >> 40) as f32 / 16777216.0 - 0.5;
                v.push(val);
            }
            vectors.push(v);
        }
        vectors
    }

    #[test]
    fn build_small_index() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let index = HnswIndex::build(&vectors, 4, 16);
        assert_eq!(index.len(), 4);
        assert_eq!(index.dimension(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn search_exact_match() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let index = HnswIndex::build(&vectors, 4, 16);
        let results = index.search(&[1.0, 0.0], 1, 16);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // exact match
        assert!(results[0].1 < 1e-6); // distance ~0
    }

    #[test]
    fn search_k_neighbors() {
        let vectors = make_random_vectors(50, 8, 42);
        let index = HnswIndex::build(&vectors, 8, 32);
        let results = index.search(&vectors[0], 5, 32);
        assert_eq!(results.len(), 5);
        // First result should be the query itself
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn search_returns_correct_count() {
        let vectors = make_random_vectors(20, 4, 123);
        let index = HnswIndex::build(&vectors, 4, 16);
        let results = index.search(&vectors[5], 3, 16);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!((d - 1.0).abs() < 1e-6); // orthogonal vectors = cosine distance 1

        let c = vec![1.0, 0.0];
        let d_same = compute_distance(&a, &c, DistanceMetric::Cosine);
        assert!(d_same < 1e-6); // same direction = cosine distance 0
    }

    #[test]
    fn l2_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = compute_distance(&a, &b, DistanceMetric::L2);
        assert!((d - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn cosine_index_build_and_search() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let index = HnswIndex::build_with_metric(&vectors, 4, 16, DistanceMetric::Cosine);
        let results = index.search(&[1.0, 0.0, 0.0], 2, 16);
        assert_eq!(results.len(), 2);
        // The closest should be vector 0 (exact) or vector 1 (very similar)
        assert!(results[0].0 == 0 || results[0].0 == 1);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let vectors = make_random_vectors(30, 4, 999);
        let index = HnswIndex::build(&vectors, 4, 16);

        let bytes = index.to_hdf5_bytes().unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");

        let loaded = HnswIndex::load_from_hdf5(&bytes, "").unwrap();
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.dimension(), index.dimension());
        assert_eq!(loaded.metric(), DistanceMetric::L2);
        assert_eq!(loaded.m, index.m);
        assert_eq!(loaded.ef_construction, index.ef_construction);
        assert_eq!(loaded.entry_point, index.entry_point);

        // Verify vectors match
        for i in 0..loaded.len() {
            assert_eq!(loaded.vectors[i], index.vectors[i]);
        }
    }

    #[test]
    fn save_and_load_preserves_search_results() {
        let vectors = make_random_vectors(40, 6, 777);
        let index = HnswIndex::build(&vectors, 6, 24);

        let query = &vectors[10];
        let original_results = index.search(query, 5, 24);

        let bytes = index.to_hdf5_bytes().unwrap();
        let loaded = HnswIndex::load_from_hdf5(&bytes, "").unwrap();
        let loaded_results = loaded.search(query, 5, 24);

        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, load) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, load.0);
            assert!((orig.1 - load.1).abs() < 1e-6);
        }
    }

    #[test]
    fn cosine_roundtrip() {
        let vectors = make_random_vectors(20, 3, 555);
        let index =
            HnswIndex::build_with_metric(&vectors, 4, 16, DistanceMetric::Cosine);

        let bytes = index.to_hdf5_bytes().unwrap();
        let loaded = HnswIndex::load_from_hdf5(&bytes, "").unwrap();
        assert_eq!(loaded.metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn index_metadata() {
        let vectors = make_random_vectors(10, 5, 111);
        let index = HnswIndex::build(&vectors, 3, 12);
        assert_eq!(index.len(), 10);
        assert_eq!(index.dimension(), 5);
        assert!(index.num_layers() >= 1);
        assert_eq!(index.metric(), DistanceMetric::L2);
    }

    #[test]
    fn save_to_file_writer() {
        let vectors = make_random_vectors(15, 3, 333);
        let index = HnswIndex::build(&vectors, 4, 16);

        let dir = std::env::temp_dir();
        let path = dir.join("rustyhdf5_ann_test_save.h5");
        let mut writer = IoFileWriter::create(&path).unwrap();
        index.save_to_hdf5(&mut writer).unwrap();

        // Verify file exists and has HDF5 signature
        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[..8], b"\x89HDF\r\n\x1a\n");

        // Load it back
        let loaded = HnswIndex::load_from_hdf5(&data, "").unwrap();
        assert_eq!(loaded.len(), 15);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn search_accuracy_l2() {
        // Build a small index and verify that brute-force search agrees
        let vectors = make_random_vectors(100, 8, 42);
        let index = HnswIndex::build(&vectors, 16, 64);

        let query = &vectors[50];
        let k = 10;
        let results = index.search(query, k, 64);

        // Brute-force nearest neighbors
        let mut brute: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, compute_distance(query, v, DistanceMetric::L2)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        brute.truncate(k);

        // With high ef and m, HNSW should find at least 80% of true neighbors
        let hnsw_ids: HashSet<usize> = results.iter().map(|r| r.0).collect();
        let brute_ids: HashSet<usize> = brute.iter().map(|r| r.0).collect();
        let overlap = hnsw_ids.intersection(&brute_ids).count();
        assert!(
            overlap >= k * 8 / 10,
            "HNSW recall too low: {overlap}/{k}"
        );
    }

    #[test]
    fn search_accuracy_cosine() {
        let vectors = make_random_vectors(100, 8, 99);
        let index =
            HnswIndex::build_with_metric(&vectors, 16, 64, DistanceMetric::Cosine);

        let query = &vectors[25];
        let k = 10;
        let results = index.search(query, k, 64);

        let mut brute: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, compute_distance(query, v, DistanceMetric::Cosine)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        brute.truncate(k);

        let hnsw_ids: HashSet<usize> = results.iter().map(|r| r.0).collect();
        let brute_ids: HashSet<usize> = brute.iter().map(|r| r.0).collect();
        let overlap = hnsw_ids.intersection(&brute_ids).count();
        assert!(
            overlap >= k * 8 / 10,
            "HNSW cosine recall too low: {overlap}/{k}"
        );
    }

    #[test]
    fn distance_metric_str_roundtrip() {
        assert_eq!(DistanceMetric::from_str("l2"), Some(DistanceMetric::L2));
        assert_eq!(
            DistanceMetric::from_str("cosine"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(DistanceMetric::from_str("unknown"), None);
        assert_eq!(DistanceMetric::L2.as_str(), "l2");
        assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let d = compute_distance(&a, &b, DistanceMetric::Cosine);
        assert!((d - 1.0).abs() < 1e-6); // zero vector -> distance 1
    }
}
