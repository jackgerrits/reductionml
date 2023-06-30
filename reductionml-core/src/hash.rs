#[inline(always)]
pub fn hash_bytes(key: &[u8], seed: u32) -> u32 {
    // murmurhash3_32(key, seed)
    twox_hash::xxh3::hash64_with_seed(key, seed as u64) as u32
}

// fn fmix(mut h: u32) -> u32 {
//     h ^= h >> 16;
//     h = h.wrapping_mul(0x85eb_ca6b);
//     h ^= h >> 13;
//     h = h.wrapping_mul(0xc2b2_ae35);
//     h ^= h >> 16;
//     h
// }

// #[allow(arithmetic_overflow)]
// fn murmurhash3_32(key: &[u8], seed: u32) -> u32 {
//     let num_blocks = key.len() / 4;
//     let mut h1 = seed;

//     const CONSTANT1: u32 = 0xcc9e_2d51;
//     const CONSTANT2: u32 = 0x1b87_3593;

//     // Body
//     for i in 0..num_blocks {
//         let mut b = Cursor::new(&key[(i * 4)..(i * 4) + 4]);
//         let mut current_block = b.read_u32::<LittleEndian>().unwrap();

//         current_block = current_block.wrapping_mul(CONSTANT1);
//         current_block = current_block.rotate_left(15);
//         current_block = current_block.wrapping_mul(CONSTANT2);

//         h1 ^= current_block;
//         h1 = h1.rotate_left(13);
//         h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
//     }

//     // Tail
//     let mut k1: u32 = 0;
//     let tail = &key[num_blocks * 4..];

//     if !tail.is_empty() {
//         if tail.len() >= 3 {
//             k1 ^= u32::from(tail[2]).wrapping_shl(16);
//         }

//         if tail.len() >= 2 {
//             k1 ^= u32::from(tail[1]).wrapping_shl(8);
//         }

//         k1 ^= u32::from(tail[0]);
//         k1 = k1.wrapping_mul(CONSTANT1);
//         k1 = k1.rotate_left(15);
//         k1 = k1.wrapping_mul(CONSTANT2);
//         h1 ^= k1;
//     }

//     // Finalization
//     h1 ^= key.len() as u32;
//     fmix(h1)
// }

pub(crate) const FNV_PRIME: u32 = 16777619;

// // Test truth values calculated using C++ implementation.
// #[test]
// fn fmix_tests() {
//     assert_eq!(fmix(0), 0);
//     assert_eq!(fmix(1), 1364076727);
//     assert_eq!(fmix(5), 3423425485);
//     assert_eq!(fmix(2147483647), 4190899880);
//     assert_eq!(fmix(4294967295), 2180083513);
// }

// #[test]
// fn hash_tests_zero_seed() {
//     assert_eq!(murmurhash3_32(b"t", 0), 3397902157);
//     assert_eq!(murmurhash3_32(b"te", 0), 3988319771);
//     assert_eq!(murmurhash3_32(b"tes", 0), 196677210);
//     assert_eq!(murmurhash3_32(b"test", 0), 3127628307);
//     assert_eq!(murmurhash3_32(b"tested", 0), 2247989476);
//     assert_eq!(
//         murmurhash3_32(b"8hv20cjwicnsj vw m000'.'.][][]...!!@3", 0),
//         4212741639
//     );
// }

// #[test]
// fn hash_tests_nonzero_seed() {
//     assert_eq!(murmurhash3_32(b"t", 25436347), 960607349);
//     assert_eq!(murmurhash3_32(b"te", 25436347), 2834341637);
//     assert_eq!(murmurhash3_32(b"tes", 25436347), 1163171263);
//     assert_eq!(murmurhash3_32(b"tested", 25436347), 3592599130);
//     assert_eq!(
//         murmurhash3_32(b"8hv20cjwicnsj vw m000'.'.][][]...!!@3", 25436347),
//         2503360452
//     );
// }

// #[test]
// fn hash_feature_tests() {
//     // Hashes calculated using VW CLI
//     assert_eq!(
//         hash_feature(
//             &features::Feature::Simple {
//                 namespace: "myNamespace".to_string(),
//                 name: "feature".to_string()
//             },
//             0
//         ),
//         1717770527
//     );
//     assert_eq!(
//         hash_feature(
//             &features::Feature::Simple {
//                 namespace: "a".to_string(),
//                 name: "a1".to_string()
//             },
//             0
//         ),
//         2579875658
//     );
//     assert_eq!(
//         hash_feature(
//             &features::Feature::SimpleWithStringValue {
//                 namespace: "myNamespace".to_string(),
//                 name: "feature".to_string(),
//                 value: "value".to_string()
//             },
//             0
//         ),
//         3812705603
//     );
//     assert_eq!(
//         hash_feature(
//             &features::Feature::Anonymous {
//                 namespace: "anon".to_string(),
//                 offset: 0
//             },
//             0
//         ),
//         659962185
//     );
//     assert_eq!(
//         hash_feature(
//             &features::Feature::Anonymous {
//                 namespace: "anon".to_string(),
//                 offset: 1
//             },
//             0
//         ),
//         659962186
//     );
// }

// #[test]
// fn hash_feature_with_bit_mask_tests() {
//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Simple {
//                     namespace: "myNamespace".to_string(),
//                     name: "feature".to_string()
//                 },
//                 0
//             ),
//             bit_mask(18)
//         ),
//         203039
//     );
//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Simple {
//                     namespace: "myNamespace".to_string(),
//                     name: "feature".to_string()
//                 },
//                 0
//             ),
//             bit_mask(5)
//         ),
//         31
//     );

//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::SimpleWithStringValue {
//                     namespace: "myNamespace".to_string(),
//                     name: "feature".to_string(),
//                     value: "value".to_string()
//                 },
//                 0
//             ),
//             bit_mask(18)
//         ),
//         83267
//     );
//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::SimpleWithStringValue {
//                     namespace: "myNamespace".to_string(),
//                     name: "feature".to_string(),
//                     value: "value".to_string()
//                 },
//                 0
//             ),
//             bit_mask(5)
//         ),
//         3
//     );

//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Anonymous {
//                     namespace: "anon".to_string(),
//                     offset: 0
//                 },
//                 0
//             ),
//             bit_mask(18)
//         ),
//         145737
//     );
//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Anonymous {
//                     namespace: "anon".to_string(),
//                     offset: 1
//                 },
//                 0
//             ),
//             bit_mask(18)
//         ),
//         145738
//     );

//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Anonymous {
//                     namespace: "anon".to_string(),
//                     offset: 0
//                 },
//                 0
//             ),
//             bit_mask(5)
//         ),
//         9
//     );
//     assert_eq!(
//         mask_hash(
//             hash_feature(
//                 &features::Feature::Anonymous {
//                     namespace: "anon".to_string(),
//                     offset: 1
//                 },
//                 0
//             ),
//             bit_mask(5)
//         ),
//         10
//     );
// }

// #[test]
// fn hash_interactions() {
//     assert_eq!(
//         hash_feature(
//             &features::Feature::Interacted {
//                 terms: vec![
//                     features::Feature::Simple {
//                         namespace: "a".to_string(),
//                         name: "a1".to_string()
//                     },
//                     features::Feature::Simple {
//                         namespace: "b".to_string(),
//                         name: "b1".to_string()
//                     }
//                 ]
//             },
//             0
//         ),
//         1046402606
//     );
// }
