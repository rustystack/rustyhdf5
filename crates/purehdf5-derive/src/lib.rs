//! Proc macros for deriving HDF5 compound type mapping.
//!
//! Provides `#[derive(H5Type)]` which generates methods for mapping Rust structs
//! to HDF5 compound datatypes, including serialization and deserialization.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

/// Derive macro that generates HDF5 compound type mapping for structs.
///
/// Generates three methods:
/// - `hdf5_datatype()` — returns the HDF5 `Datatype::Compound` descriptor
/// - `to_bytes(&self)` — serializes the struct to HDF5 compound raw bytes
/// - `from_bytes(data: &[u8])` — deserializes from HDF5 compound raw bytes
///
/// # Supported field types
/// - `f32`, `f64`
/// - `i8`, `i16`, `i32`, `i64`
/// - `u8`, `u16`, `u32`, `u64`
/// - `bool` (stored as `u8`)
/// - `[T; N]` fixed-size arrays of any supported numeric type
#[proc_macro_derive(H5Type)]
pub fn derive_h5type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_h5type(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_h5type(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "H5Type can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "H5Type can only be derived for structs",
            ));
        }
    };

    let mut datatype_member_stmts = Vec::new();
    let mut serialize_stmts = Vec::new();
    let mut deserialize_stmts = Vec::new();
    let mut field_names = Vec::new();
    let mut size_increments = Vec::new();

    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_name_str = field_name.to_string();
        let ty = &field.ty;

        let (dt_expr, ser_expr, deser_expr, size_expr) = type_mapping(ty, field_name)?;

        datatype_member_stmts.push(quote! {
            _members.push(purehdf5_format::datatype::CompoundMember {
                name: #field_name_str.into(),
                byte_offset: _offset,
                datatype: #dt_expr,
            });
            _offset += #size_expr as u64;
        });

        size_increments.push(quote! { + (#size_expr as usize) });
        serialize_stmts.push(ser_expr);
        deserialize_stmts.push(deser_expr);
        field_names.push(field_name.clone());
    }

    let expanded = quote! {
        impl #name {
            /// Returns the HDF5 compound datatype descriptor for this struct.
            pub fn hdf5_datatype() -> purehdf5_format::datatype::Datatype {
                let mut _offset: u64 = 0;
                let mut _members = Vec::new();
                #(#datatype_member_stmts)*
                purehdf5_format::datatype::Datatype::Compound {
                    size: _offset as u32,
                    members: _members,
                }
            }

            /// Serializes this struct to HDF5 compound raw bytes (little-endian).
            pub fn to_bytes(&self) -> Vec<u8> {
                let mut _buf = Vec::with_capacity(Self::_h5_compound_size());
                #(#serialize_stmts)*
                _buf
            }

            /// Deserializes from HDF5 compound raw bytes (little-endian).
            pub fn from_bytes(_data: &[u8]) -> Self {
                let mut _pos = 0usize;
                #(#deserialize_stmts)*
                Self {
                    #(#field_names),*
                }
            }

            fn _h5_compound_size() -> usize {
                0usize #(#size_increments)*
            }
        }
    };

    Ok(expanded)
}

fn type_mapping(
    ty: &Type,
    field_name: &syn::Ident,
) -> syn::Result<(
    proc_macro2::TokenStream, // datatype expression
    proc_macro2::TokenStream, // serialize expression
    proc_macro2::TokenStream, // deserialize expression
    proc_macro2::TokenStream, // size expression
)> {
    match ty {
        Type::Path(type_path) => {
            let seg = type_path.path.segments.last().unwrap();
            let type_name = seg.ident.to_string();
            match type_name.as_str() {
                "f64" => Ok(float_mapping(field_name, 8, 64, 52, 11, 52, 1023)),
                "f32" => Ok(float_mapping(field_name, 4, 32, 23, 8, 23, 127)),
                "i8" => Ok(int_mapping(field_name, 1, true)),
                "i16" => Ok(int_mapping(field_name, 2, true)),
                "i32" => Ok(int_mapping(field_name, 4, true)),
                "i64" => Ok(int_mapping(field_name, 8, true)),
                "u8" => Ok(int_mapping(field_name, 1, false)),
                "u16" => Ok(int_mapping(field_name, 2, false)),
                "u32" => Ok(int_mapping(field_name, 4, false)),
                "u64" => Ok(int_mapping(field_name, 8, false)),
                "bool" => Ok(bool_mapping(field_name)),
                _ => Err(syn::Error::new_spanned(
                    ty,
                    format!("unsupported type `{type_name}` for H5Type derive"),
                )),
            }
        }
        Type::Array(arr) => {
            let elem_ty = &*arr.elem;
            let len_expr = &arr.len;
            array_mapping(field_name, elem_ty, len_expr)
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported type for H5Type derive",
        )),
    }
}

fn float_mapping(
    field_name: &syn::Ident,
    size: u32,
    precision: u16,
    mant_loc: u8,
    exp_size: u8,
    mant_size: u8,
    exp_bias: u32,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let size_lit = size;
    let precision_lit = precision;
    let exp_size_lit = exp_size;
    let mant_size_lit = mant_size;
    let exp_bias_lit = exp_bias;
    let exp_loc: u8 = mant_loc;

    let dt = quote! {
        purehdf5_format::datatype::Datatype::FloatingPoint {
            size: #size_lit,
            byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: #precision_lit,
            exponent_location: #exp_loc,
            exponent_size: #exp_size_lit,
            mantissa_location: 0,
            mantissa_size: #mant_size_lit,
            exponent_bias: #exp_bias_lit,
        }
    };

    let ser = quote! {
        _buf.extend_from_slice(&self.#field_name.to_le_bytes());
    };

    let deser = if size == 8 {
        quote! {
            let #field_name = f64::from_le_bytes(
                _data[_pos.._pos + 8].try_into().unwrap()
            );
            _pos += 8;
        }
    } else {
        quote! {
            let #field_name = f32::from_le_bytes(
                _data[_pos.._pos + 4].try_into().unwrap()
            );
            _pos += 4;
        }
    };

    let sz = size as usize;
    let size_expr = quote! { #sz };
    (dt, ser, deser, size_expr)
}

fn int_mapping(
    field_name: &syn::Ident,
    size: u32,
    signed: bool,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let precision = (size * 8) as u16;

    let dt = quote! {
        purehdf5_format::datatype::Datatype::FixedPoint {
            size: #size,
            byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
            signed: #signed,
            bit_offset: 0,
            bit_precision: #precision,
        }
    };

    let ser = quote! {
        _buf.extend_from_slice(&self.#field_name.to_le_bytes());
    };

    let sz = size as usize;
    let deser = match (size, signed) {
        (1, true) => quote! {
            let #field_name = _data[_pos] as i8;
            _pos += 1;
        },
        (1, false) => quote! {
            let #field_name = _data[_pos];
            _pos += 1;
        },
        (2, true) => quote! {
            let #field_name = i16::from_le_bytes(
                _data[_pos.._pos + 2].try_into().unwrap()
            );
            _pos += 2;
        },
        (2, false) => quote! {
            let #field_name = u16::from_le_bytes(
                _data[_pos.._pos + 2].try_into().unwrap()
            );
            _pos += 2;
        },
        (4, true) => quote! {
            let #field_name = i32::from_le_bytes(
                _data[_pos.._pos + 4].try_into().unwrap()
            );
            _pos += 4;
        },
        (4, false) => quote! {
            let #field_name = u32::from_le_bytes(
                _data[_pos.._pos + 4].try_into().unwrap()
            );
            _pos += 4;
        },
        (8, true) => quote! {
            let #field_name = i64::from_le_bytes(
                _data[_pos.._pos + 8].try_into().unwrap()
            );
            _pos += 8;
        },
        (8, false) => quote! {
            let #field_name = u64::from_le_bytes(
                _data[_pos.._pos + 8].try_into().unwrap()
            );
            _pos += 8;
        },
        _ => quote! {
            let mut _tmp = [0u8; #sz];
            _tmp.copy_from_slice(&_data[_pos.._pos + #sz]);
            let #field_name = _tmp;
            _pos += #sz;
        },
    };

    let sz = size as usize;
    let size_expr = quote! { #sz };
    (dt, ser, deser, size_expr)
}

fn bool_mapping(
    field_name: &syn::Ident,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    let dt = quote! {
        purehdf5_format::datatype::Datatype::FixedPoint {
            size: 1,
            byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
    };

    let ser = quote! {
        _buf.push(if self.#field_name { 1u8 } else { 0u8 });
    };

    let deser = quote! {
        let #field_name = _data[_pos] != 0;
        _pos += 1;
    };

    let size_expr = quote! { 1usize };
    (dt, ser, deser, size_expr)
}

fn array_mapping(
    field_name: &syn::Ident,
    elem_ty: &Type,
    len_expr: &syn::Expr,
) -> syn::Result<(
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
)> {
    let Type::Path(type_path) = elem_ty else {
        return Err(syn::Error::new_spanned(
            elem_ty,
            "array element must be a primitive type for H5Type derive",
        ));
    };
    let elem_name = type_path.path.segments.last().unwrap().ident.to_string();

    let (base_dt, elem_size, deser_one) = match elem_name.as_str() {
        "f64" => (
            quote! {
                purehdf5_format::datatype::Datatype::FloatingPoint {
                    size: 8,
                    byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
                    bit_offset: 0, bit_precision: 64,
                    exponent_location: 52, exponent_size: 11,
                    mantissa_location: 0, mantissa_size: 52,
                    exponent_bias: 1023,
                }
            },
            8usize,
            quote! { f64::from_le_bytes(_data[_pos.._pos + 8].try_into().unwrap()) },
        ),
        "f32" => (
            quote! {
                purehdf5_format::datatype::Datatype::FloatingPoint {
                    size: 4,
                    byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
                    bit_offset: 0, bit_precision: 32,
                    exponent_location: 23, exponent_size: 8,
                    mantissa_location: 0, mantissa_size: 23,
                    exponent_bias: 127,
                }
            },
            4usize,
            quote! { f32::from_le_bytes(_data[_pos.._pos + 4].try_into().unwrap()) },
        ),
        "i8" => (
            int_dt_quote(1, true),
            1usize,
            quote! { _data[_pos] as i8 },
        ),
        "i16" => (
            int_dt_quote(2, true),
            2usize,
            quote! { i16::from_le_bytes(_data[_pos.._pos + 2].try_into().unwrap()) },
        ),
        "i32" => (
            int_dt_quote(4, true),
            4usize,
            quote! { i32::from_le_bytes(_data[_pos.._pos + 4].try_into().unwrap()) },
        ),
        "i64" => (
            int_dt_quote(8, true),
            8usize,
            quote! { i64::from_le_bytes(_data[_pos.._pos + 8].try_into().unwrap()) },
        ),
        "u8" => (
            int_dt_quote(1, false),
            1usize,
            quote! { _data[_pos] },
        ),
        "u16" => (
            int_dt_quote(2, false),
            2usize,
            quote! { u16::from_le_bytes(_data[_pos.._pos + 2].try_into().unwrap()) },
        ),
        "u32" => (
            int_dt_quote(4, false),
            4usize,
            quote! { u32::from_le_bytes(_data[_pos.._pos + 4].try_into().unwrap()) },
        ),
        "u64" => (
            int_dt_quote(8, false),
            8usize,
            quote! { u64::from_le_bytes(_data[_pos.._pos + 8].try_into().unwrap()) },
        ),
        _ => {
            return Err(syn::Error::new_spanned(
                elem_ty,
                format!("unsupported array element type `{elem_name}` for H5Type derive"),
            ));
        }
    };

    let dt = quote! {
        purehdf5_format::datatype::Datatype::Array {
            base_type: Box::new(#base_dt),
            dimensions: vec![#len_expr as u32],
        }
    };

    let ser = quote! {
        for _elem in &self.#field_name {
            _buf.extend_from_slice(&_elem.to_le_bytes());
        }
    };

    let deser = quote! {
        let #field_name = {
            let mut _arr = [Default::default(); #len_expr];
            for _i in 0..#len_expr {
                _arr[_i] = #deser_one;
                _pos += #elem_size;
            }
            _arr
        };
    };

    let size_expr = quote! { (#len_expr * #elem_size) };
    Ok((dt, ser, deser, size_expr))
}

fn int_dt_quote(size: u32, signed: bool) -> proc_macro2::TokenStream {
    let precision = (size * 8) as u16;
    quote! {
        purehdf5_format::datatype::Datatype::FixedPoint {
            size: #size,
            byte_order: purehdf5_format::datatype::DatatypeByteOrder::LittleEndian,
            signed: #signed,
            bit_offset: 0,
            bit_precision: #precision,
        }
    }
}
