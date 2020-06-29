// use std::fs::File;
// use std::io::Write;

use kiss3d::scene::SceneNode;

static TYPES: [&str; 3] = ["in", "liq", "out"];


#[derive(Clone)]
pub struct Neuron {
    // maybe it should know which cluster it belongs to? and cluster could be a
    obj: SceneNode, // a sphere design associated with the neuron
    connects: Vec<u32>,
    spec: String, // specialization
    // v: f32, // voltage input
    // theta: f32, // threshold to activate
    // v_rest: f32, // resting voltage
    // n_t : String,
    // read_out: bool,
}

impl Neuron {
    pub fn new(
        obj: SceneNode,
        connects: Vec<u32>, /*, v: f32, theta: f32, v_rest: f32, n_t: String, input: bool, read_out: bool*/
        spec: &str,
    ) -> Neuron {
        assert!(true, TYPES.contains(&spec));
        Self {
            obj: obj,
            connects: connects,
            spec: spec.to_string(), // either "liq", "in", or "out"
            // v: v,
            // theta: theta,
            // v_rest: v_rest
            // n_t: n_t
            // input: input,
            // read_out: read_out
        }
    }

    pub fn get_obj(&mut self) -> &mut SceneNode {
        &mut self.obj
    }

    pub fn set_connects(&mut self, connects: Vec<u32>) {
        self.connects = connects;
    }

    pub fn get_connects(&self) -> &Vec<u32> {
        &self.connects
    }

    pub fn set_spec(&mut self, id: &str) {
        assert!(true, TYPES.contains(&id));
        self.spec = id.to_string();
    }

    pub fn get_spec(&self) -> &String {
        & self.spec
    }
}