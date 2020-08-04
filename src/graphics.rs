use kiss3d::camera::ArcBall;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::scene::SceneNode;
use kiss3d::window::Window;

use nalgebra::geometry::Translation3;
use nalgebra::Point3;

// mod machine;
use crate::machine::LSM;

pub fn render_lines(
    window: &mut Window,                                          // Our window
    axis_len: f32,                                                // The length of the axis lines
    lines: &Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>,         // The edges between neurons
    dists: &Vec<f32>, // Distances between each neuron to each other neuron
    readout_lines: &Vec<(Point3<f32>, Point3<f32>, Point3<f32>)>, // Very similar to lines vector above
    n_total: usize,                                               // Number of Liquid Neurons
    time_steps: u32,
    conditions: [bool; 8],
    l: &mut LSM, // List of neurons
) {
    // Renders the edges between neurons as well as the lines of axis. \\

    // Cloning global variables so we can make them mutable.
    let mut axis_on: bool = conditions[0];
    let mut lines_on: bool = conditions[1];
    let mut readout_lines_on: bool = conditions[2];
    let mut fancy: bool = conditions[3];
    let mut yaw: bool = conditions[4];
    let mut sp_on: bool = conditions[5];
    let mut readout_on: bool = conditions[6];
    let rm_dis_n = conditions[7];

    // We want to start off at a point other than the origin so we don't have to
    // zoom out immediately.
    let eye = Point3::new(10.0, 10.0, 10.0);
    // let eye = Point3::new(7., 7., 7.);
    let at = Point3::origin();
    let mut arc_ball = ArcBall::new(eye, at);

    // Removes neurons that are not connected to any other neurons
    if rm_dis_n {
        l.remove_disconnects(window);
    }

    // Connect spheres will be filled of a line of spheres in a gradient
    // of white to black
    let mut connect_sp: Vec<SceneNode> = Vec::new();
    if fancy {
        connect_sp = fancy_lines(window, lines, dists);
    }

    // let mut spike_times: Vec<Vec<u32>> = Vec::new();
    // for n_idx in 0..n_total {
    //     spike_times.push(l.get_spike_times(n_idx).clone());
    // }

    // thread::spawn(move || {
    //     for curr_t in 0..time_steps {
    //         for n_idx in 0..n_total {
    //             let b = spike_times;
    //         }
    //     }
    // });

    let mut time_step = 0;
    // let mut neurons = l.get_neurons();
    // Arc ball allows for some useful user controls.
    while window.render_with_camera(&mut arc_ball) {
        // update the current camera.
        for event in window.events().iter() {
            // When a key is pressed, put it into key
            match event.value {
                WindowEvent::Key(key, Action::Release, _) => {
                    // Toggle the axis with A
                    if key == Key::A {
                        axis_on = !axis_on;
                    // Toggle the edges between liquid neurons with L
                    } else if key == Key::L {
                        lines_on = !lines_on;
                    // Toggle the readout lines with C
                    } else if key == Key::C {
                        readout_lines_on = !readout_lines_on;
                    }
                    // Toggle the yaw with Y
                    else if key == Key::Y {
                        yaw = !yaw;
                    // Toggle the liquid neurons with S
                    } else if key == Key::S {
                        sp_on = !sp_on;
                    // Toggle the readout set with R
                    } else if key == Key::R {
                        readout_on = !readout_on;
                    }
                    // Toggle fancy lines with F (causes lag with big setups)
                    else if key == Key::F {
                        fancy = !fancy;
                    }
                }
                _ => {}
            }
        }

        for n_idx in 0..n_total {
            let speed = 5;
            let spike_len = 5;
            if time_step / speed > spike_len {
                let past_spike_time = l.get_neurons()[n_idx]
                    .get_spike_times()
                    .contains(&((time_step / speed) - spike_len));
                if past_spike_time {
                    l.get_neurons()[n_idx].get_obj().set_color(
                        172. / 255.,
                        222. / 255.,
                        248. / 255.,
                    );
                }
            }
            let curr_spike_time = l.get_neurons()[n_idx]
                .get_spike_times()
                .contains(&(time_step / speed));
            // println!("{}", l.get_neurons()[n_idx].get_spike_times().is_empty());
            if curr_spike_time {
                // l.get_neurons()[n_idx].get_obj().set_visible(false);
                l.get_neurons()[n_idx].get_obj().set_color(1., 0., 0.);
            }

            if time_step / speed + spike_len + 1 > time_steps {
                time_step = 0;
            }
        }

        if !sp_on {
            sp_show(l);
            sp_on = !sp_on;
        }

        if !readout_on {
            readout_show(l);
            readout_on = !readout_on;
        }

        if !fancy {
            fancy_show(&mut connect_sp);
            fancy = !fancy;
        }

        if yaw {
            // A yaw rotates the whole window slowly
            let curr_yaw = arc_ball.yaw();
            arc_ball.set_yaw(curr_yaw + 0.025);
        }

        // Our axis lines
        // x axis - Red
        // y axis - Green
        // z axis - Blue
        if axis_on {
            window.draw_line(
                &Point3::new(-axis_len, 0.0, 0.0),
                &Point3::new(axis_len, 0.0, 0.0),
                &Point3::new(1.0, 0.0, 0.0),
            );
            window.draw_line(
                &Point3::new(0.0, -axis_len, 0.0),
                &Point3::new(0.0, axis_len, 0.0),
                &Point3::new(0.0, 1.0, 0.0),
            );
            window.draw_line(
                &Point3::new(0.0, 0.0, -axis_len),
                &Point3::new(0.0, 0.0, axis_len),
                &Point3::new(0.0, 0.0, 1.0),
            );
        }

        // Draw edges between neurons
        if lines_on {
            for coords in lines.iter() {
                window.draw_line(&coords.0, &coords.1, &coords.2);
            }
        }

        // Lines between liquid neurons and the readout set of neurons
        if readout_lines_on {
            for coords in readout_lines.iter() {
                window.draw_line(&coords.0, &coords.1, &coords.2);
            }
        }
        time_step += 1;
    }
}

fn fancy_lines(
    window: &mut Window, // We draw the spheres directly using the window
    lines: &Vec<(
        Point3<f32>, // Start point of a line
        Point3<f32>, // Endpoint of a line
        Point3<f32>, // R, G, B color values
    )>,
    dists: &Vec<f32>, // Distance of one sphere to each other in order of LSM.neurons
) -> Vec<SceneNode> {
    // Alternative way to draw all the lines in the graphics. We represent lines by small spheres. \\
    let sp_per_dist = 0.03_f32; // 1 sphere every 0.05 units
    let mut spheres: Vec<SceneNode> = Vec::new(); // Our output
    for idx in 0..lines.len() {
        let n_spheres: usize = (dists[idx] / sp_per_dist) as usize; // number of spheres in 1 line
                                                                    // Coordinates of each point
        let x1 = (lines[idx].0)[0];
        let y1 = (lines[idx].0)[1];
        let z1 = (lines[idx].0)[2];
        let x2 = (lines[idx].1)[0];
        let y2 = (lines[idx].1)[1];
        let z2 = (lines[idx].1)[2];

        // Drawing 1 line made of 'n_spheres' spheres
        for k in 0..n_spheres {
            let k = (k as f32) / (n_spheres as f32); // shadowing it to do a range of floats
            let mut sp = window.add_sphere(0.01);

            // (x1, y1, z1) + k(x2 - x1, y2 - y1, z2 - z1) is each new sphere
            // center since k increments a bit each time.
            let t = Translation3::new(x1 + k * (x2 - x1), y1 + k * (y2 - y1), z1 + k * (z2 - z1));
            sp.append_translation(&t);
            // Color changes from white to black
            sp.set_color(1.0 - k, 1.0 - k, 1.0 - k);
            // We don't care about moving the actual sphere at this point
            spheres.push(sp);
        }
    }
    spheres
}

fn sp_show(l: &mut LSM) {
    // Toggles the visibility of neurons in an LSM \\
    let neurons = l.get_neurons();
    // We don't want to assert, we want it to print regardless.
    if neurons.len() == 0 {
        return;
    }
    // If one is visible, then they are all visible
    let visible: bool = neurons[0].get_obj().is_visible();
    for n_idx in 0..neurons.len() {
        // get_obj() gets a reference to the sphere in a neuron
        neurons[n_idx].get_obj().set_visible(!visible);
    }
}

fn readout_show(l: &mut LSM) {
    // Toggles the visibility of readout neurons. \\
    let readouts = l.get_readouts();
    let visible: bool = readouts[0][0].get_obj().is_visible();
    // Readouts is a list of clusters (clusters are lists of neurons).
    for cluster_idx in 0..readouts.len() {
        for readout in readouts[cluster_idx].iter_mut() {
            // get_obj returns a mutable reference to a Sphere
            readout.get_obj().set_visible(!visible);
        }
    }
}

fn fancy_show(sp_list: &mut Vec<SceneNode>) {
    // Toggles the visibility of any set of spheres. \\
    if sp_list.len() == 0 {
        return;
    }

    // If one is visible, then they are all visible
    let visible: bool = sp_list[0].is_visible();
    for sp_idx in 0..sp_list.len() {
        sp_list[sp_idx].set_visible(!visible);
    }
}
