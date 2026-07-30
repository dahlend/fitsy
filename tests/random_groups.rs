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

/// Standard Sec.6.1.2: `physical = PZEROn + PSCALn x stored`, and
/// repeated `PTYPEn` names name one parameter whose value is the sum of
/// the slots. Modelled on the UVFITS pattern where a Julian date is
/// split into a coarse and a fine `DATE` slot so the parameter carries
/// more precision than `BITPIX` alone allows.
#[test]
fn group_parameters_apply_pscal_pzero_and_sum_repeated_ptype() {
    let cards = [
        pad_card("SIMPLE  =                    T"),
        pad_card("BITPIX  =                  -32"),
        pad_card("NAXIS   =                    2"),
        pad_card("NAXIS1  =                    0"),
        pad_card("NAXIS2  =                    1"),
        pad_card("GROUPS  =                    T"),
        pad_card("PCOUNT  =                    3"),
        pad_card("GCOUNT  =                    1"),
        pad_card("PTYPE1  = 'UU      '"),
        pad_card("PSCAL1  =                 2.0"),
        pad_card("PZERO1  =                 0.5"),
        // Two slots share the name DATE: a whole-day part carried at
        // full scale and a fraction-of-day part scaled down.
        pad_card("PTYPE2  = 'DATE    '"),
        pad_card("PSCAL2  =                 1.0"),
        pad_card("PZERO2  =         2400000.5"),
        pad_card("PTYPE3  = 'DATE    '"),
        pad_card("PSCAL3  =               0.125"),
        pad_card("PZERO3  =                 0.0"),
        pad_card("END"),
    ];
    let mut buf: Vec<u8> = Vec::new();
    for c in &cards {
        buf.extend_from_slice(c);
    }
    pad_to_block(&mut buf, b' ');
    // params [3.0, 100.0, 2.0], one data value.
    for v in [3.0_f32, 100.0, 2.0, 42.0] {
        buf.extend_from_slice(&v.to_be_bytes());
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::RandomGroups(rg) = f.hdu(0).unwrap() else {
        panic!("expected RandomGroups HDU");
    };

    // Per-slot physical values: eq. 6.1 applied slot by slot.
    let phys = rg.group_parameters(0).unwrap();
    assert_eq!(phys, vec![0.5 + 2.0 * 3.0, 2400000.5 + 100.0, 0.125 * 2.0]);

    // Defaults: a slot with no PSCALn/PZEROn is an identity mapping.
    let params = rg.parameters();
    assert_eq!(params.len(), 3);
    assert_eq!(params[0].name, "UU");
    assert_eq!(params[1].pscal, 1.0);
    assert_eq!(params[2].pzero, 0.0);

    // Repeated PTYPEn: the two DATE slots sum into one parameter.
    let date = rg.group_parameter_by_name(0, "DATE").unwrap().unwrap();
    assert_eq!(date, (2400000.5 + 100.0) + 0.125 * 2.0);
    // A name appearing once is just its own physical value.
    assert_eq!(rg.group_parameter_by_name(0, "UU").unwrap(), Some(6.5));
    // Repeats collapse to a single logical parameter.
    assert_eq!(rg.parameter_names(), vec!["UU", "DATE"]);
    assert_eq!(rg.group_parameter_by_name(0, "VV").unwrap(), None);
    assert!(rg.group_parameters(1).is_err());
}

/// Sec.6.1.2 scaling must be independent of `BITPIX`: an integer-typed
/// random-groups HDU is the case where `PSCALn`/`PZEROn` matter most.
#[test]
fn group_parameters_scale_integer_bitpix() {
    let cards = [
        pad_card("SIMPLE  =                    T"),
        pad_card("BITPIX  =                   16"),
        pad_card("NAXIS   =                    2"),
        pad_card("NAXIS1  =                    0"),
        pad_card("NAXIS2  =                    1"),
        pad_card("GROUPS  =                    T"),
        pad_card("PCOUNT  =                    1"),
        pad_card("GCOUNT  =                    1"),
        pad_card("PTYPE1  = 'WW      '"),
        pad_card("PSCAL1  =              1.0E-3"),
        pad_card("PZERO1  =                -1.0"),
        pad_card("END"),
    ];
    let mut buf: Vec<u8> = Vec::new();
    for c in &cards {
        buf.extend_from_slice(c);
    }
    pad_to_block(&mut buf, b' ');
    for v in [-32768_i16, 7] {
        buf.extend_from_slice(&v.to_be_bytes());
    }
    pad_to_block(&mut buf, 0);

    let f = FitsFile::from_bytes(buf).unwrap();
    let Hdu::RandomGroups(rg) = f.hdu(0).unwrap() else {
        panic!("expected RandomGroups HDU");
    };
    let phys = rg.group_parameters(0).unwrap();
    assert!(
        (phys[0] - (-1.0 + 1.0e-3 * -32768.0)).abs() < 1e-12,
        "got {phys:?}"
    );
}
