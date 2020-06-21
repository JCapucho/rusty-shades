global in=0 a_position: Vector<3, Float>;
global in=1 a_color: Vector<3, Float>;
global in=2 a_normal: Vector<3, Float>;
        
global out=0 v_position: Vector<3, Float>;
global out=1 v_color: Vector<3, Float>;
global out=2 v_normal: Vector<3, Float>;

global position gl_position;

global uniform=(set=0,binding=0) globals: Globals;
global uniform=(set=2,binding=0) locals: Locals;

struct Globals {
    u_view_proj: Matrix<4, 4, Float>,
    u_view_position: Vector<3, Float>
}

struct Locals {
    u_transform: Matrix<4, 4, Float>,
    u_min_max: Vector<2, Float>
}
        
fn vertex main() {
    v_color = a_color;
    v_normal = a_normal;
        
    v_position = (u_transform * vec4(a_position, 1.0)).xyz;
    gl_Position = u_view_proj * u_transform * vec4(a_position, 1.0);
}