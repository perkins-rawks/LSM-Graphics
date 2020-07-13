// use std::fs::File;
// use std::io::Write;

use kiss3d::scene::SceneNode;
use nalgebra::base::Vector3;
use nalgebra::geometry::Translation3;
use std::fmt;

static TYPES: [&str; 4] = ["in", "liq_in", "liq", "readout"]; // The three types of neurons in our LSM
pub static NTS: [&str; 2] = ["exc", "inh"]; // types of neurotransmitters

#[derive(Clone)]
pub struct Neuron {
    id: usize,
    obj: SceneNode, // a sphere design associated with the neuron
    // connects: Vec<u32>, // 1's and 0's representing a connection for 1 and not for 0
    spec: String,                 // Specialization (either input, output, or liquid)
    nt: String,                   // Neurotransmitter type, excitatory or inhibitory
    cluster: String,              // The cluster self belongs to. Usually a task name.
    loc: Vector3<f32>,            // The location of the sphere associated with self
    input_weight: i32,            // The weight at which spike trains are read (only for "liq_in")
    v: f32,                       // Voltage of a neuron (mV)
    v_th: f32,                    // Voltage threshold to activate (mV)
    v_rest: f32,                  // The resting voltage (mV)
    refrac_period: u32, // Refractory period, the time (in ms) it takes to get back to resting
    time_out: u32,      // Used for tracking refractory period
    pre_syn_connects: Vec<usize>, // Indices of pre-synaptic neurons
    spike_times: Vec<u32>, // The times at which a spike / neuron fire happens (an index of a 1 in a spike train)
    input_connect: usize, // The index of the input layer neuron this is connected to. (only if "liq_in")
    second_tau: [u32; 2], // Time constants for second order model for voltage
    c: f32,                       // Calcium of a readout neuron
    c_th: f32,                    // Calcium threshold for a readout neuron
    c_d: f32,                     // Desired calcium of a readout neuron
    c_margin: f32,                // Margin around the C threshold
}

impl fmt::Display for Neuron {
    // Printing a Neuron \\
    // Formatter: '_ is a lifetime argument making it so that fmt lasts as long
    // as self
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.spec == "readout".to_string() {
            write!(
                f,
                "Neuron: {{ id: {} -- type: \"{}\" -- nt: \"{}\" -- voltage: {} mV -- calcium: {} }}",
                self.id, self.spec, self.nt, self.v, self.c
            )
        } else {
            write!(
                f,
                "Neuron: {{ id: {} -- type: \"{}\" -- nt: \"{}\" -- voltage: {} mV }}",
                self.id, self.spec, self.nt, self.v
            )
        }
    }
}

impl Neuron {
    // Constructor for Neuron objects \\
    pub fn new(
        obj: SceneNode,
        // connects: Vec<u32>, /*, v: f32, theta: f32, v_rest: f32, n_t: String, input: bool, read_out: bool*/
        spec: &str,
        cluster: &str,
    ) -> Neuron {
        assert!(true, TYPES.contains(&spec)); // has to be inside TYPES
        Self {
            id: 0,
            obj: obj,
            // connects: connects,
            spec: spec.to_string(),
            // n_t: n_t
            // input: input,
            // read_out: read_out
            nt: "".to_string(), // neither exc nor inh
            cluster: cluster.to_string(),
            loc: Vector3::<f32>::new(0., 0., 0.),
            input_weight: 0,
            v: 0.,
            v_th: 20.,
            v_rest: 0.,
            pre_syn_connects: Vec::new(),
            spike_times: Vec::new(),
            input_connect: 0,
            refrac_period: 2,
            time_out: 0,
            second_tau: [0, 0],
            c: 0.,
            c_th: 5.,
            c_d: 0.,
            c_margin: 3.,
            //tau_m: 32,
            //tau_c: 64,
        }
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    pub fn get_obj(&mut self) -> &mut SceneNode {
        &mut self.obj
    }

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

    pub fn get_loc(&self) -> &Vector3<f32> {
        // & so we don't move it
        &self.loc
    }

    pub fn set_input_weight(&mut self, weight: i32) {
        // moving value
        self.input_weight = weight;
    }

    pub fn set_pre_syn_connects(&mut self, indices: Vec<usize>) {
        // moving value
        self.pre_syn_connects = indices;
    }

    pub fn get_pre_syn_connects(&self) -> &Vec<usize> {
        &self.pre_syn_connects
    }

    pub fn get_spike_times(&self) -> &Vec<u32> {
        // & so we don't move it
        &self.spike_times
    }

    pub fn set_spike_times(&mut self, spike_times: Vec<u32>) {
        // moving value
        self.spike_times = spike_times;
    }

    pub fn set_input_connect(&mut self, input_idx: usize) {
        self.input_connect = input_idx;
    }

    pub fn get_input_connect(&self) -> usize {
        // We are not moving it, so we clone
        self.input_connect.clone()
    }

    pub fn get_input_weight(&self) -> i32 {
        // We are not moving it, so we clone
        self.input_weight.clone()
    }

    pub fn get_voltage(&self) -> f32 {
        self.v.clone()
    }

    pub fn update_voltage(&mut self, delta_v: f32) {
        // Moves the voltage value
        self.v += delta_v;
    }

    pub fn get_calcium(&self) -> f32 {
        self.c.clone()
    }

    pub fn update_calcium(&mut self, delta_c: f32) {
        // Moves the calcium value
        self.c += delta_c;
    }

    pub fn set_calcium_desired(&mut self, new_c_d: f32) {
        // Moves parameter c_d
        self.c_d = new_c_d;
    }

    pub fn get_calcium_desired(&self) {
        &self.c_d;
    }

    pub fn update_spike_times(&mut self, curr_t: usize) {
        // Updates the spike times based on voltage. \\
        // If the current voltage is greater than threshold,
        // add the spike time to self attribute and update voltage
        if self.v_th < self.v {
            self.spike_times.push(curr_t as u32);
            self.time_out = self.refrac_period;
            self.v = -5.; // so that it's harder to spike after resting
        }
    }

    pub fn get_time_out(&self) -> u32 {
        self.time_out
    }

    pub fn update_time_out(&mut self) {
        assert_eq!(self.time_out > 0, true);
        self.time_out -= 1;
    }

    pub fn get_second_tau(&self) -> &[u32; 2] {
        &self.second_tau
    }

    pub fn set_second_tau(&mut self, tau_s1: u32, tau_s2: u32) {
        self.second_tau = [tau_s1, tau_s2];
    }

    pub fn update_calcium_desired(&self) {

    }

    // // Temporary
    // pub fn update_calcium_desired (&mut self) {
    //     self.c_d = 3.*self.c_th;
    // }
}
