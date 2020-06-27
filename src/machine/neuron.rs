// use std::fs::File;
// use std::io::Write;

use kiss3d::scene::SceneNode;


#[derive(Clone)]
pub struct Neuron {
    // maybe it should know which cluster it belongs to? and cluster could be a
    pub obj: SceneNode, // a sphere design associated with the neuron
    pub connects: Vec<u32>,
    // v: f32, // voltage input
    // theta: f32, // threshold to activate
    // v_rest: f32, // resting voltage
    // n_t : String,
    // input: bool,
    // read_out: bool,
}

impl Neuron {
    pub fn new(
        obj: SceneNode,
        connects: Vec<u32>, /*, v: f32, theta: f32, v_rest: f32, n_t: String, input: bool, read_out: bool*/
    ) -> Neuron {
        Self {
            obj: obj,
            connects: connects,
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
}