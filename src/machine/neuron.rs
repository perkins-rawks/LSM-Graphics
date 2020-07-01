// use std::fs::File;
// use std::io::Write;

use kiss3d::scene::SceneNode;
use nalgebra::base::Vector3;
use nalgebra::geometry::Translation3;
use std::fmt;

static TYPES: [&str; 3] = ["in", "liq", "out"]; // The three types of neurons in our LSM
pub static NTS: [&str; 2] = ["exc", "inh"]; // types of neurotransmitters

#[derive(Clone)]
pub struct Neuron {
    // maybe it should know which cluster it belongs to? and cluster could be a
    obj: SceneNode,     // a sphere design associated with the neuron
    // connects: Vec<u32>, // 1's and 0's representing a connection for 1 and not for 0
    spec: String,       // specialization
    nt: String,         // neurotransmitter type, from the choice of NTS
    loc: Vector3<f32>,
    // input: Vec<u32>,    // a spike train. 
                        // v: f32, // voltage input
                        // theta: f32, // threshold to activate
                        // v_rest: f32, // resting voltage
                        // n_t : String,
                        // read_out: bool,
}

impl fmt::Display for Neuron {
    // Printing a Neuron \\
    // Formatter: '_ is a lifetime argument making it so that fmt lasts as long
    // as self 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neuron: {} and {}", self.get_spec(), self.get_nt())
    }
}

impl Neuron {
    // Constructor for Neuron objects \\
    pub fn new(
        obj: SceneNode,
        // connects: Vec<u32>, /*, v: f32, theta: f32, v_rest: f32, n_t: String, input: bool, read_out: bool*/
        spec: &str,
    ) -> Neuron {
        assert!(true, TYPES.contains(&spec)); // has to be inside TYPES
        Self {
            obj: obj,
            // connects: connects,
            spec: spec.to_string(),
            // v: v,
            // theta: theta,
            // v_rest: v_rest
            // n_t: n_t
            // input: input,
            // read_out: read_out
            nt: "".to_string(), // neither exc nor inh
            loc: Vector3::<f32>::new(0., 0., 0.),

        }
    }

    pub fn get_obj(&mut self) -> &mut SceneNode {
        &mut self.obj
    }

    // pub fn set_connects(&mut self, connects: Vec<u32>) {
    //     self.connects = connects;
    // }

    pub fn set_spec(&mut self, id: &str) {
        assert!(true, TYPES.contains(&id));
        self.spec = id.to_string();
    }

    pub fn get_spec(&self) -> &String {
        &self.spec
    }

    pub fn get_nt(&self) -> &String {
        &self.nt
    }

    pub fn set_nt(&mut self, nt: &str) {
        assert_eq!(true, NTS.contains(&nt));
        self.nt = nt.to_string();
    }

    pub fn set_loc(&mut self, loc: &Translation3<f32>) {
        self.loc = loc.vector;
    }

    pub fn get_loc(&self) -> &Vector3::<f32> {
        &self.loc
    }
}
