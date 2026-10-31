function mat_out = exportar_matrices_control_octave(pole_scale, obs_scale, out_mat_path)
%EXPORTAR_MATRICES_CONTROL_OCTAVE Sintetiza el modelo linealizado, las pruebas de Kalman,
% el PID descentralizado en espacio articular y la realimentacion de estados con accion
% integral + observador de Luenberger, y los exporta para el simulador interactivo.

engine_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(engine_dir, 'config'));
addpath(fullfile(engine_dir, 'kinematics'));
addpath(fullfile(engine_dir, 'dynamics'));
addpath(fullfile(engine_dir, 'solvers'));
addpath(fullfile(engine_dir, 'controllers'));

if nargin < 1 || isempty(pole_scale), pole_scale = 1.0; end
if nargin < 2 || isempty(obs_scale),  obs_scale  = 1.0; end
if nargin < 3 || isempty(out_mat_path)
    out_mat_path = fullfile(engine_dir, '..', 'results', 'octave_control_matrices.mat');
end

P = stewart_params();
sys_nom = linealizacion_analitica('mechanical', P);

opts = struct('pole_speed_factor', double(pole_scale), 'obs_speed_factor', double(obs_scale));
ctrl_pi = controlador_estado_observador(P, sys_nom, 'integral_pi', opts);
ctrl_pid_leg = controlador_pid(P, sys_nom, 'leg_space');

mat_out = struct();
mat_out.Rb = P.Rb;
mat_out.Rp = P.Rp;
mat_out.h0 = P.h0;
mat_out.B_pts = P.B;
mat_out.A_pts = P.A;
mat_out.pose0 = P.pose0;
mat_out.L0 = P.L0;
mat_out.Lmin = P.Lmin;
mat_out.Lmax = P.Lmax;
mat_out.lowerBodyLength = P.lowerBodyLength;
mat_out.massTop = P.massTop;
mat_out.massRod = P.massRod;
mat_out.platformInertiaBody = P.platformInertiaBody;
mat_out.gravity = P.gravity;          % parametros de las patas para la planta no lineal del simulador
mat_out.legMC = P.leg.mC; mat_out.legMS = P.leg.mS;
mat_out.legBC = P.leg.bC; mat_out.legBS = P.leg.bS;
mat_out.legIC = P.leg.IC; mat_out.legIS = P.leg.IS;
[M0, ~, ~, ~, J0, T0] = stewart_dynamic_matrices(P.pose0, zeros(6,1), P);
mat_out.M0 = M0;           % matriz de inercia nominal (trayectoria variacional 6 GDL)
mat_out.J0 = J0 * T0;      % Jacobiano nominal dl/dq
mat_out.Fmax = P.act.Fmax;
mat_out.Ts = sys_nom.Ts;

% Matrices del sistema linealizado y verificacion de Kalman en GNU Octave
mat_out.A = sys_nom.A;
mat_out.B = sys_nom.B;
mat_out.C = sys_nom.C;
mat_out.D = sys_nom.D;
mat_out.Ad = sys_nom.Ad;
mat_out.Bd = sys_nom.Bd;
mat_out.Cd = sys_nom.Cd;
mat_out.x0 = sys_nom.x0;
mat_out.u0 = sys_nom.u0;
mat_out.rank_Mc = sys_nom.rank_Mc;
mat_out.rank_Mo = sys_nom.rank_Mo;

% Realimentacion de estados con accion integral + observador de Luenberger
mat_out.K_pi = ctrl_pi.K;
mat_out.Ki_pi = ctrl_pi.Ki;
mat_out.Nr_pi = ctrl_pi.Nr;
mat_out.L_cont = ctrl_pi.L;
mat_out.Ld = ctrl_pi.Ld;
mat_out.err_Nr_cof = ctrl_pi.err_Nr_cof;
mat_out.sep_error = ctrl_pi.sep_error;
mat_out.poles_cl = ctrl_pi.poles_cl;
mat_out.poles_obs = ctrl_pi.poles_obs;
mat_out.obs_poles_disc = ctrl_pi.obs_poles_disc;
mat_out.obs_poles_impl = ctrl_pi.obs_poles_impl;
mat_out.pole_scale = pole_scale;
mat_out.obs_scale = obs_scale;

% Ganancias de diseno del PID descentralizado en espacio articular
mat_out.Kp_leg = ctrl_pid_leg.Kp;
mat_out.Ki_leg = ctrl_pid_leg.Ki;
mat_out.Kd_leg = ctrl_pid_leg.Kd;

save('-v7', out_mat_path, '-struct', 'mat_out');
fprintf('Matrices de control exportadas a: %s\n', out_mat_path);
end
