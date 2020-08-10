/*
    We process the input file coming from lsm-input program in this file. 
*/

#![allow(dead_code)]

use std::fs;

pub fn read_input() -> Vec<Vec<u32>> {
    // Read our input file (a list of spike trains)
    // This input is of the form (# columns > # rows usually)
    // 0 0 0 1 1 0 0 1 0
    // 0 1 0 1 1 0 1 0 1
    // ...
    // This function stores that in a list by column.
    const RADIX: u32 = 10; // base 10
    let contents = fs::read_to_string("input.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut input: Vec<Vec<u32>> = Vec::new();
    for line in contents.iter() {
        let mut mini_input: Vec<u32> = Vec::new(); // One line
        let new_line = line.trim(); // Trims whitespace
        for character in new_line.chars() {
            if character != ' ' {
                // to_digit takes a number base
                mini_input.push(character.to_digit(RADIX).unwrap());
            }
        }
        input.push(mini_input);
    }
    input
}

pub fn read_labels() -> Vec<String> {
    // A label is a description of the action desired for each column of input.
    // Our file, labels.txt, is what we will use to supervise learning.
    let contents = fs::read_to_string("labels.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut labels: Vec<String> = Vec::new();
    for line in contents.iter() {
        labels.push(line.trim().to_string());
    }
    labels
}

pub fn read_training_input() -> Vec<Vec<Vec<u32>>> {
    // Read the input file (a list of spike trains).
    // This input is of the form where # rows > # columns and there
    // is a '#' marking the end of an epoch.
    // 0 0 0 1 1 0 0 1 0
    // 0 1 0 1 1 0 1 0 1
    // ...
    // #
    // ...
    // #
    // This function stores that in a list by column.
    let mut input: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut train_batch: Vec<Vec<u32>> = Vec::new();

    const RADIX: u32 = 10; // base 10
    let contents = fs::read_to_string("train_input.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();

    for line in contents.iter() {
        let new_line = line.trim(); // Trims whitespace
        if new_line == "#" {
            input.push(train_batch);
            train_batch = Vec::new();
            continue;
        }
        let mut mini_input: Vec<u32> = Vec::new(); // One line
        for character in new_line.chars() {
            if character != ' ' {
                // to_digit takes a number base
                mini_input.push(character.to_digit(RADIX).unwrap());
            }
        }
        train_batch.push(mini_input);
    }
    input
}

pub fn read_training_labels() -> Vec<String> {
    // A label is a description of the action desired for each column of input.
    // Our file, labels.txt, is what we will use to supervise learning.
    // Labels.txt will look like this, where each line is an epoch of labels.
    // run
    // eat
    // ...
    let contents = fs::read_to_string("train_labels.txt").expect("Unable to read input file");
    let contents: Vec<&str> = contents.split("\n").collect();
    let mut labels: Vec<String> = Vec::new();
    for line in contents.iter() {
        let new_line = line.trim(); // Trims whitespace
        labels.push(new_line.to_string());
    }
    labels
}