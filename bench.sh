#!/bin/bash
cargo build && sudo perf record -g target/debug/software_rasterizer &&  sudo perf report