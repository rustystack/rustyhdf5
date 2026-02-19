//! HDF5 Shared Object Header Message resolution.
//!
//! When a header message has its "shared" flag (bit 1 of msg_flags) set,
//! the message data is not the actual message content but a reference
//! to a shared copy stored elsewhere.
//!
//! Shared message reference types:
//! - Type 0: shared in the same object header (not typically used)
//! - Type 1: shared in another object header (version 1-2)
//! - Type 2: shared in the SOHM table (via fractal heap, version 3)
//! - Type 3: shared in another object header (version 3)

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::FormatError;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;

/// A resolved shared message reference.
#[derive(Debug, Clone)]
pub struct SharedMessageRef {
    /// The type of shared message reference.
    pub ref_type: u8,
    /// Version of the shared message encoding.
    pub version: u8,
    /// Address of the object header containing the shared message (type 1, 3).
    pub object_header_address: Option<u64>,
}

fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    if pos + s > data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: pos + s,
            available: data.len(),
        });
    }
    Ok(match size {
        2 => u16::from_le_bytes([data[pos], data[pos + 1]]) as u64,
        4 => u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64,
        8 => u64::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

fn ensure_len(data: &[u8], pos: usize, needed: usize) -> Result<(), FormatError> {
    if pos + needed > data.len() {
        Err(FormatError::UnexpectedEof {
            expected: pos + needed,
            available: data.len(),
        })
    } else {
        Ok(())
    }
}

/// Check whether a header message has its shared flag set.
pub fn is_shared(msg_flags: u8) -> bool {
    msg_flags & 0x02 != 0
}

/// Parse a shared message reference from the message data.
///
/// When the shared flag is set on a message, the data contains a reference
/// instead of the actual message content.
pub fn parse_shared_ref(
    data: &[u8],
    offset_size: u8,
) -> Result<SharedMessageRef, FormatError> {
    ensure_len(data, 0, 2)?;
    let version = data[0];
    let ref_type = data[1];

    match version {
        1 | 2 => {
            // v1/v2: reserved(6) + address(offset_size)
            let pos = 2 + 6; // skip reserved bytes
            ensure_len(data, pos, offset_size as usize)?;
            let addr = read_offset(data, pos, offset_size)?;
            Ok(SharedMessageRef {
                ref_type,
                version,
                object_header_address: Some(addr),
            })
        }
        3 => {
            match ref_type {
                1 | 3 => {
                    // type 1/3: message in another object header
                    // v3 layout: version(1) + type(1) + address(offset_size)
                    ensure_len(data, 2, offset_size as usize)?;
                    let addr = read_offset(data, 2, offset_size)?;
                    Ok(SharedMessageRef {
                        ref_type,
                        version,
                        object_header_address: Some(addr),
                    })
                }
                2 => {
                    // type 2: SOHM table (fractal heap ID)
                    // We store the address as None since this needs heap resolution
                    Ok(SharedMessageRef {
                        ref_type,
                        version,
                        object_header_address: None,
                    })
                }
                _ => Err(FormatError::InvalidSharedMessageVersion(ref_type)),
            }
        }
        _ => Err(FormatError::InvalidSharedMessageVersion(version)),
    }
}

/// Resolve a shared message to its actual message data.
///
/// For type 1/3 (shared in another object header), reads the target object header
/// and finds the message of the specified type.
pub fn resolve_shared_message(
    file_data: &[u8],
    shared_ref: &SharedMessageRef,
    target_msg_type: MessageType,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    match shared_ref.ref_type {
        1 | 3 => {
            let addr = shared_ref.object_header_address.ok_or(
                FormatError::UnexpectedEof {
                    expected: 1,
                    available: 0,
                }
            )?;
            let target_header =
                ObjectHeader::parse(file_data, addr as usize, offset_size, length_size)?;
            for msg in &target_header.messages {
                if msg.msg_type == target_msg_type && !is_shared(msg.flags) {
                    return Ok(msg.data.clone());
                }
            }
            // The message at that OH address is the message itself
            // In many cases with type 1, the entire OH at that address IS the shared message
            // Try returning the first message of any type that isn't Nil
            for msg in &target_header.messages {
                if msg.msg_type == target_msg_type {
                    return Ok(msg.data.clone());
                }
            }
            // Fall back to first non-nil message
            for msg in &target_header.messages {
                if msg.msg_type != MessageType::Nil {
                    return Ok(msg.data.clone());
                }
            }
            Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            })
        }
        _ => {
            // Type 0 (same OH) and type 2 (SOHM heap) are not yet supported
            Err(FormatError::InvalidSharedMessageVersion(shared_ref.ref_type))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_shared_flag() {
        assert!(!is_shared(0x00));
        assert!(!is_shared(0x01));
        assert!(is_shared(0x02));
        assert!(is_shared(0x03));
        assert!(is_shared(0x06));
    }

    #[test]
    fn parse_v3_type1_ref() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(1); // type 1 = shared in another OH
        data.extend_from_slice(&0x1234u64.to_le_bytes()); // address

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 1);
        assert_eq!(shared.object_header_address, Some(0x1234));
    }

    #[test]
    fn parse_v3_type3_ref() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(3); // type 3 = shared in another OH (v3 encoding)
        data.extend_from_slice(&0xABCDu64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 3);
        assert_eq!(shared.object_header_address, Some(0xABCD));
    }

    #[test]
    fn parse_v1_ref() {
        let mut data = Vec::new();
        data.push(1); // version
        data.push(0); // type
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&0x5678u64.to_le_bytes());

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 1);
        assert_eq!(shared.object_header_address, Some(0x5678));
    }

    #[test]
    fn parse_v2_ref() {
        let mut data = Vec::new();
        data.push(2); // version
        data.push(0); // type
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&0x9000u32.to_le_bytes());

        let shared = parse_shared_ref(&data, 4).unwrap();
        assert_eq!(shared.version, 2);
        assert_eq!(shared.object_header_address, Some(0x9000));
    }

    #[test]
    fn parse_v3_type2_sohm() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(2); // type 2 = SOHM heap

        let shared = parse_shared_ref(&data, 8).unwrap();
        assert_eq!(shared.version, 3);
        assert_eq!(shared.ref_type, 2);
        assert_eq!(shared.object_header_address, None);
    }

    #[test]
    fn invalid_version() {
        let data = vec![99, 0];
        let err = parse_shared_ref(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidSharedMessageVersion(99));
    }

    #[test]
    fn truncated_data() {
        let data = vec![3u8]; // too short
        let err = parse_shared_ref(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_four_byte_offsets() {
        let mut data = Vec::new();
        data.push(3); // version
        data.push(1); // type 1
        data.extend_from_slice(&0x1000u32.to_le_bytes());

        let shared = parse_shared_ref(&data, 4).unwrap();
        assert_eq!(shared.object_header_address, Some(0x1000));
    }
}
