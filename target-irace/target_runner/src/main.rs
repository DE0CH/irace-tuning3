use std::collections::HashMap;
use itertools::Itertools;
use itertools::EitherOrBoth::*;
use std::fs;
use serde_json;
use std::io::Read;
use std::io::{stdout, Write};
use curl::easy::{Easy, List};

fn main() {
    let mut b: Vec<String> = Vec::new();
    let a = [
        "", // configuration_id
        "", // instance_id
        "seed",
        "instance_name",
        "", // bound_max
    ].into_iter().zip_longest(std::env::args().into_iter().skip(1));
    let mut m: HashMap<_, String> = a.filter_map::<(&str, String), _>(|x| {match x {
        Both(l, r) => {
            if l != "" {
                Some((l, r.into()))
            } else {
                None
            }
        },
        Left(_) => panic!("insufficient arguments"),
        Right(r) => {b.push(r); None},
    }}).collect();
    m.insert("instance_name", String::from(&m["instance_name"][2..]));
    m.insert("dir", ".".to_string());
    let mut m: HashMap<_, serde_json::Value> = m.into_iter().map(|(k, v)| (k, v.into())).collect();
    m.insert("cutoff", i64::MAX.into()); //TODO: figure how to to disable timeout
    m.insert("seed", m["seed"].as_str().unwrap().parse::<i64>().unwrap().into());
    m.insert("params", b.into());
    m.insert("instance_info", serde_json::Value::Null);
    m.insert("run_length", serde_json::Value::Null);
    let s = serde_json::to_string(&m).unwrap();
    let host = fs::read_to_string("nameserver_creds/host.txt")
        .expect("No host file found");
    let port = fs::read_to_string("nameserver_creds/port.txt")
        .expect("No port file found");
    let url = format!("http://{}:{}/predict-simple", host, port);

    let mut data = s.as_bytes();

    let mut easy = Easy::new();
    easy.url(&url).unwrap();
    let mut list = List::new();
    list.append("Content-Type: application/json").unwrap();
    easy.http_headers(list).unwrap();
    easy.post(true).unwrap();
    easy.post_field_size(data.len() as u64).unwrap();

    let mut transfer = easy.transfer();
    transfer.read_function(|buf| {
        Ok(data.read(buf).unwrap_or(0))
    }).unwrap();
    transfer.write_function(|data| {
        stdout().write_all(data).unwrap();
        Ok(data.len())
    }).unwrap();
    transfer.perform().unwrap();
}
