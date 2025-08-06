// ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName

use std::io::BufRead;

pub struct PerfData {
    pub id: u32,
    pub op_type: String,
    pub data_type: String,
    pub target: String,
    pub input_shape: String,
    pub output_shape: String,
    pub cycles: String,
    pub time: String,
    pub mac_usage: String,
    pub work_load: String,
    pub rw: String,
    pub full_name: String,
}

impl PerfData {
    pub fn new(
        id: u32,
        op_type: String,
        data_type: String,
        target: String,
        input_shape: String,
        output_shape: String,
        cycles: String,
        time: String,
        mac_usage: String,
        work_load: String,
        rw: String,
        full_name: String,
    ) -> Self {
        PerfData {
            id,
            op_type,
            data_type,
            target,
            input_shape,
            output_shape,
            cycles,
            time,
            mac_usage,
            work_load,
            rw,
            full_name,
        }
    }
}

/// A simple fixed‐width parser based on a header line.
pub struct TableParser {
    /// Byte indices where each column begins
    starts: Vec<usize>,
}

impl TableParser {
    /// Build a parser by giving it the exact header line and a list of column names
    /// in the order they appear.
    ///
    /// # Panics
    /// Panics if any `column_names` entry isn’t found in `header`.
    pub fn new(header: &str, column_names: &[&str]) -> Self {
        // 1) find start indices
        let mut starts = Vec::with_capacity(column_names.len());
        for &name in column_names {
            if let Some(idx) = header.find(name) {
                starts.push(idx);
            } else {
                panic!("column `{}` not found in header line", name);
            }
        }

        TableParser { starts }
    }

    /// Finds all non-whitespace "words" in a line and returns them with their
    /// starting byte index.
    fn find_words_with_indices(line: &str) -> Vec<(usize, &str)> {
        let mut words = Vec::new();
        let mut start_index = 0;
        let mut in_word = false;

        // We manually iterate to track indices correctly
        for (i, char) in line.char_indices() {
            if !char.is_whitespace() && !in_word {
                // Found the start of a new word
                start_index = i;
                in_word = true;
            } else if char.is_whitespace() && in_word {
                // Found the end of the word
                words.push((start_index, &line[start_index..i]));
                in_word = false;
            }
        }

        // If the line ends with a word, we need to capture it
        if in_word {
            words.push((start_index, &line[start_index..]));
        }

        words
    }

    // In impl TableParser

    /// Given a data line, return a Vec of Options—one per column—by aligning
    /// found words to the nearest preceding header position.
    pub fn parse_line<'a>(&self, line: &'a str) -> Vec<Option<String>> {
        // Step 1: Find all data words and their start indices.
        let words = Self::find_words_with_indices(line);
        let mut word_iter = words.iter().peekable();

        let mut result: Vec<Option<String>> = vec![None; self.starts.len()];

        // Step 2: Iterate through each column definition to align the words.
        for (i, &col_start) in self.starts.iter().enumerate() {
            // Peek at the next available word
            if let Some((word_start, word)) = word_iter.peek() {
                // The logic: a word belongs to column `i` if it starts before
                // the header of the *next* column (`i + 1`).
                let next_col_start = self.starts.get(i + 1).cloned().unwrap_or(usize::MAX);

                if *word_start < next_col_start {
                    // This word belongs to the current column. Assign it.
                    result[i] = Some(word.to_string());
                    // Consume the word so it's not considered for the next column.
                    word_iter.next();
                }
                // If the condition is false, it means this column is empty. We do
                // nothing and `result[i]` remains `None`. The same word will be
                // checked against the next column.
            }
        }

        result
    }
}

pub fn parse_perf_data<R: BufRead>(input: R) -> (Vec<PerfData>, Option<SummaryStats>) {
    let mut perf_data = Vec::new();
    let parser = TableParser::new(
        "ID   OpType             DataType Target InputShape                               OutputShape            Cycles(DDR/NPU/Total)    Time(us)     MacUsage(%)          WorkLoad(0/1/2)      RW(KB)       FullName",
        &[
            "ID",
            "OpType",
            "DataType",
            "Target",
            "InputShape",
            "OutputShape",
            "Cycles(DDR/NPU/Total)",
            "Time(us)",
            "MacUsage(%)",
            "WorkLoad(0/1/2)",
            "RW(KB)",
            "FullName",
        ],
    );

    let input = input.lines().skip(5);

    let mut hit_summary = false;

    let mut extra_lines = Vec::new();

    for line in input {
        if let Ok(line) = line {
            if line.starts_with("-") {
                hit_summary = true;
                continue;
            }
            if hit_summary {
                extra_lines.push(line);
                continue;
            }

            let columns = parser.parse_line(&line); // Returns Vec<Option<String>>

            perf_data.push(PerfData {
                // Use and_then to safely parse, providing a default on failure.
                id: columns
                    .get(0)
                    .and_then(|opt| opt.as_ref())
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                // Use cloned() or map() for optional strings.
                op_type: columns
                    .get(1)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                data_type: columns
                    .get(2)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                target: columns
                    .get(3)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                input_shape: columns
                    .get(4)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                output_shape: columns
                    .get(5)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                cycles: columns
                    .get(6)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                time: columns
                    .get(7)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                mac_usage: columns
                    .get(8)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                work_load: columns
                    .get(9)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                rw: columns
                    .get(10)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
                full_name: columns
                    .get(11)
                    .and_then(|opt| opt.clone())
                    .unwrap_or_default(),
            });
        }
    }
    (perf_data, parse_summary(&extra_lines))
}

#[derive(Debug)]
pub struct SummaryStats {
    pub total_operator_time_us: u32,
    pub total_memory_rw_kb: f32,
    pub op_time_ranking: Vec<OpTimeRank>,
    pub total_cpu_time_us: u32,
    pub total_gpu_time_us: u32,
    pub total_npu_time_us: u32,
    pub total_all_time_us: u32,
}

#[derive(Debug)]
pub struct OpTimeRank {
    pub op_type: String,
    pub call_count: u32,
    pub cpu_time_us: u32,
    pub gpu_time_us: u32,
    pub npu_time_us: u32,
    pub total_time_us: u32,
    pub time_ratio_pct: f32,
}

fn parse_summary(lines: &[String]) -> Option<SummaryStats> {
    let mut total_operator_time_us = 0;
    let mut total_memory_rw_kb = 0.0;
    let mut op_time_ranking = Vec::new();

    for line in lines {
        if line.starts_with("Total Operator Elapsed") {
            if let Some(val) = line.split(':').nth(1) {
                total_operator_time_us = val.trim().parse().unwrap_or(0);
            }
        } else if line.starts_with("Total Memory Read/Write") {
            if let Some(val) = line.split(':').nth(1) {
                total_memory_rw_kb = val.trim().parse().unwrap_or(0.0);
            }
        } else if line.starts_with("Total") && !line.contains("Operator") {
            // Parse total row at bottom of table
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                let total_cpu_time_us = parts[1].parse().unwrap_or(0);
                let total_gpu_time_us = parts[2].parse().unwrap_or(0);
                let total_npu_time_us = parts[3].parse().unwrap_or(0);
                let total_all_time_us = parts[4].parse().unwrap_or(0);
                return Some(SummaryStats {
                    total_operator_time_us,
                    total_memory_rw_kb,
                    op_time_ranking,
                    total_cpu_time_us,
                    total_gpu_time_us,
                    total_npu_time_us,
                    total_all_time_us,
                });
            }
        } else if line.starts_with("---") || line.trim().is_empty() || line.contains("OpType") {
            continue;
        } else {
            // Parse a row in the time ranking table
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 7 {
                op_time_ranking.push(OpTimeRank {
                    op_type: parts[0].to_string(),
                    call_count: parts[1].parse().unwrap_or(0),
                    cpu_time_us: parts[2].parse().unwrap_or(0),
                    gpu_time_us: parts[3].parse().unwrap_or(0),
                    npu_time_us: parts[4].parse().unwrap_or(0),
                    total_time_us: parts[5].parse().unwrap_or(0),
                    time_ratio_pct: parts[6].trim_end_matches('%').parse().unwrap_or(0.0),
                });
            }
        }
    }

    None
}
