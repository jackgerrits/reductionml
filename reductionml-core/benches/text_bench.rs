use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use reductionml_core::{
    self,
    object_pool::{Pool, PoolReturnable},
    parsers::{TextModeParser, TextModeParserFactory},
};

pub fn parser_text_float_heavy(c: &mut Criterion) {
    let pool = Arc::new(Pool::new());
    let text_parser = reductionml_core::parsers::VwTextParserFactory::default().create(
        reductionml_core::FeaturesType::SparseSimple,
        reductionml_core::LabelType::Simple,
        0,
        18,
        pool.clone(),
    );
    let input_text = "1 |f 13:3.9656971e-02 24:3.4781646e-02 69:4.6296168e-02 85:6.1853945e-02 140:3.2349996e-02 156:1.0290844e-01 175:6.8493910e-02 188:2.8366476e-02 229:7.4871540e-02 230:9.1505975e-02 234:5.4200061e-02 236:4.4855952e-02 238:5.3422898e-02 387:1.4059304e-01 394:7.5131744e-02 433:1.1118756e-01 434:1.2540409e-01 438:6.5452829e-02 465:2.2644201e-01 468:8.5926279e-02 518:1.0214076e-01 534:9.4191484e-02 613:7.0990764e-02 646:8.7701865e-02 660:7.2289191e-02 709:9.0660661e-02 752:1.0580081e-01 757:6.7965068e-02 812:2.2685185e-01 932:6.8250686e-02 1028:4.8203137e-02 1122:1.2381379e-01 1160:1.3038123e-01 1189:7.1542501e-02 1530:9.2655659e-02 1664:6.5160148e-02 1865:8.5823394e-02 2524:1.6407280e-01 2525:1.1528353e-01 2526:9.7131468e-02 2536:5.7415009e-01 2543:1.4978983e-01 2848:1.0446861e-01 3370:9.2423186e-02 3960:1.5554591e-01 7052:1.2632671e-01 16893:1.9762035e-01 24036:3.2674628e-01 24303:2.2660980e-01";
    c.bench_function("parser_text_float_heavy", |b| {
        b.iter(|| {
            let (feats, _) = text_parser.parse_chunk(black_box(input_text)).unwrap();
            feats.clear_and_return_object(&pool);
        })
    });
}

pub fn parser_text_hash_heavy(c: &mut Criterion) {
    let pool = Arc::new(Pool::new());
    let text_parser = reductionml_core::parsers::VwTextParserFactory::default().create(
        reductionml_core::FeaturesType::SparseSimple,
        reductionml_core::LabelType::Simple,
        0,
        18,
        pool.clone(),
    );

    let input_text = "1 | Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam";
    c.bench_function("parser_text_hash_heavy", |b| {
        b.iter(|| {
            let (feats, _) = text_parser.parse_chunk(black_box(input_text)).unwrap();
            feats.clear_and_return_object(black_box(&pool));
        })
    });
}

pub fn parser_text_cb(c: &mut Criterion) {
    let pool = Arc::new(Pool::new());
    let text_parser = reductionml_core::parsers::VwTextParserFactory::default().create(
        reductionml_core::FeaturesType::SparseCBAdf,
        reductionml_core::LabelType::CB,
        0,
        18,
        pool.clone(),
    );
    let input_text = r#"shared |user Tom Lorem ipsum dolor
0:-1:0.5 |action politics sit amet, consectetur
|action sports adipiscing elit, sed do"#;
    c.bench_function("parser_text_cb", |b| {
        b.iter(|| {
            let (feats, _) = text_parser.parse_chunk(black_box(input_text)).unwrap();
            feats.clear_and_return_object(&pool);
        })
    });
}

criterion_group!(
    text_benchmarks,
    parser_text_float_heavy,
    parser_text_hash_heavy,
    parser_text_cb
);
criterion_main!(text_benchmarks);
