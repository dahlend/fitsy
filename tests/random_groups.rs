//! Random Groups primary HDU (Standard Sec.6).

use fitsy::{FitsFile, Hdu};

const BLOCK: usize = 2880;

fn pad_card(s: &str) -> [u8; 80] {
    let mut c = [b' '; 80];
    c[..s.len()].copy_from_slice(s.as_bytes());
    c
}

fn pad_to_block(buf: &mut Vec<u8>, fill: u8) {
    while !buf.len().is_multiple_of(BLOCK) {
        buf.push(fill);
    }
}

#[test]
fn random_groups_round_trip() {
    // 2 groups, each with 3 parameters and a 2x2 image plane (4 data
    // values), BITPIX = -32 (f32). NAXIS = 3, NAXIS1 = 0, NAXIS2 = 2,
    // NAXIS3 = 2, PCOUNT = 3, GCOUNT = 2.
    let cards = [
        pad_card("SIMPLE  =                    T"),
        pad_card("BITPIX  =                  -32"),
        pad_card("NAXIS   =                    3"),
        pad_card("NAXIS1  =                    0"),
        pad_card("NAXIS2  =                    2"),
        pad_card("NAXIS3  =                    2"),
        pad_card("GROUPS  =                    T"),
        pad_card("PCOUNT  =                    3"),
        pad_card("GCOUNT  =                    2"),
        pad_card("END"),
    ];
    let mut buf: Vec<u8> = Vec::new();
    for c in &cards {
        buf.extend_from_slice(c);
    }
    pad_to_block(&mut buf, b' ');

    // Group 0: params [1.0,2.0,3.0], data [10,11,12,13]
    // Group 1: params [4.0,5.0,6.0], data [20,21,22,23]
    let payloads: [(Vec<f32>, Vec<f32>); 2] = [
        (vec![1.0, 2.0, 3.0], vec![10.0, 11.0, 12.0, 13.0]),
        (vec![4.0, 5.0, 6.0], vec![20.0, 21.0, 22.0, 23.0]),
    ];
    for (params, data) in &payloads {
        for &p in params {
            buf.extend_from_slice(&p.to_be_bytes());
        }
        for &d in data {
            buf.extend_from_slice(&d.to_be_bytes());
        }
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    assert_eq!(f.len(), 1);
    let Hdu::RandomGroups(rg) = f.hdu(0).unwrap() else {
        panic!("expected RandomGroups HDU");
    };
    assert_eq!(rg.n_groups(), 2);
    assert_eq!(rg.pcount(), 3);
    assert_eq!(rg.data_per_group(), 4);
    for (i, (expected_params, expected_data)) in payloads.iter().enumerate() {
        let (params, data) = rg.group_raw::<f32>(i as u64).unwrap();
        assert_eq!(&params, expected_params, "group {i} params");
        assert_eq!(&data, expected_data, "group {i} data");
    }
    assert!(rg.group_raw::<f32>(2).is_err());
}
