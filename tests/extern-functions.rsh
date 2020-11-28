global in=0 v_position: Vector<4, Float>;

global position gl_position: Vector<4, Float>;

extern fn fclamp(x: Float, min: Float, max: Float) -> Float;

fn vertex vertex_main() {
    let a = v_position.x;
    let b = v_position.y;
    let c = v_position.z;

    let scalar = fclamp(a, b, c);

	gl_position = scalar * v_position;
}

// args: --color never build --target=glsl