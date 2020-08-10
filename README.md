# LSM-Graphics
Graphical representation of a Liquid State Machine.

## About the program:
- There are a total of 7 files containing source code for this program. 
    - mod.rs is the Liquid State Machine structure code
    - neuron.rs is the Neuron structure, with each neuron in the LSM being this type
    - utils.rs is a file containing auxillary mathematical functions that are used in mod.rs
    - analysis.rs deals with basic analysis of the liquid and other small things
    - graphics.rs is code that renders and creates the graphical part of this program
    - input.rs is where we read input from our lsm-input project and send it to mod.rs
    - main.rs is our main method where we set up the simulation and run it all
- There are files that are our output for the program. 
    - analysis.txt is where some basic analysis of the liquid is printed
    - epochs.txt contains, in order of epochs, where the first LSM was correct or incorrect
    - guesses.txt has all of the guesses per epoch of what the signal is
    - l2-guesses.txt has all of the guesses per epoch by LSM1 trying to recognize LSM2 talking
    - weights.txt is the file that contains all the readout weights at any particular time step

## How to run:
- Installation tutorial for Rust at https://doc.rust-lang.org/book/ch01-01-installation.html
- Enter "cargo run" in terminal

## Graphics: 
| Name of Setting      | Setting   | RGB (If applicable) |
| -------------------- | --------- | -------------------:|
| Axis lines           | Off       |                     |
| Fancy connections    | Off       |                     |
| Input layer          | Invisible |                     |
| Liquid connection    | White     | (255, 255, 255)     |
| Liquid connections   | On        |                     |
| Liquid input neurons | Hot pink  | (241, 21, 150)      |
| Liquid neuron        | Cyan Gray | (172, 222, 248)     |
| Readout connection   | On        |                     |
| Readout connections  | Salmon    | (227, 120, 105)     |
| Readout neuron       | White     | (255, 255, 255)     |
| Readout neurons      | On        |                     |
| Sphere radius        | 0.1 units |                     |
| Spiking neuron       | Red       | (255, 0, 0)         |

## User Controls
- A - Toggle 3D Axis
- L - Toggle connections between the neurons
- Y - Toggle rotation
- S - Toggle neurons' visibility
- R - Toggle readout neurons
- C - Toggle connections between the readout and reservoir neurons
- F - Toggle fancier connections 
