% ENTREGA_1_BENCHMARK_OCTAVE.m
% Script maestro de validacion en GNU Octave para la Entrega 1:
% Modelado cinematico/dinamico, PID articular y realimentacion de estados con accion integral + Luenberger.

clear; clc; close all;

root_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(root_dir, 'config'));
addpath(fullfile(root_dir, 'kinematics'));
addpath(fullfile(root_dir, 'dynamics'));
addpath(fullfile(root_dir, 'solvers'));
addpath(fullfile(root_dir, 'controllers'));
addpath(fullfile(root_dir, 'trajectories'));

fprintf('=========================================================================\n');
fprintf('  PROYECTO FINAL: PLATAFORMA STEWART 6-GDL - BENCHMARK MOTOR GNU OCTAVE\n');
fprintf('=========================================================================\n\n');

% 1. Parametros de la plataforma y linealizacion del modelo de Lopes
P = stewart_params();
sys_nom = linealizacion_analitica('mechanical', P);

% 2. Sintesis de los dos controladores: PID descentralizado en espacio articular y realimentacion
%    de estados con accion integral + observador de Luenberger. Si existe results/ganancias_optimas.json
%    (scripts/optimizar_ganancias.py) se usan las ganancias optimas; si no, las de diseno.
opt_file = fullfile(root_dir, '..', 'results', 'ganancias_optimas.json');
pid_opts = struct();
ss_opts = struct();
if exist(opt_file, 'file')
    opt = jsondecode(fileread(opt_file));
    pid_opts = struct('Kp_scale', opt.pid.Kp_scale, 'Ki_scale', opt.pid.Ki_scale, 'Kd_scale', opt.pid.Kd_scale);
    ss_opts = struct('pole_speed_factor', opt.estado.pole_scale, 'obs_speed_factor', opt.estado.obs_scale);
    fprintf('Ganancias optimas: PID (Kp x%.3f, Ki x%.3f, Kd x%.3f); polos x%.3f, observador x%.3f\n', ...
        opt.pid.Kp_scale, opt.pid.Ki_scale, opt.pid.Kd_scale, opt.estado.pole_scale, opt.estado.obs_scale);
else
    fprintf('Sin ganancias optimas: se usan las de diseno.\n');
end
ctrl_pid   = controlador_pid(P, sys_nom, 'leg_space', pid_opts);
ctrl_ss_pi = controlador_estado_observador(P, sys_nom, 'integral_pi', ss_opts);
ctrl_pid.name = 'PID articular';
ctrl_ss_pi.name = 'Realim. estados + integral';

controllers = {ctrl_pid, ctrl_ss_pi};
n_ctrl = numel(controllers);

fprintf('\n--- Verificacion de la sintesis en espacio de estados ---\n');
fprintf('  * ||K||_F = %.4f,  ||Ki||_F = %.4f,  ||L||_F = %.4f\n', ...
    norm(ctrl_ss_pi.K, 'fro'), norm(ctrl_ss_pi.Ki, 'fro'), norm(ctrl_ss_pi.L, 'fro'));
fprintf('  * Error cofactores ||Nr - Nr_cof||_rel:  %.3e\n', ctrl_ss_pi.err_Nr_cof);
fprintf('  * Error principio de separacion (24x24): %.3e\n\n', ctrl_ss_pi.sep_error);

% 3. Evaluacion sobre trayectorias representativas del protocolo
selected_trajs = [1, 4, 6];
trajs = cell(1, 7);
for tr_id = selected_trajs
    trajs{tr_id} = generar_trayectorias_test(tr_id, 6.0, [], P);
end
results_e1 = struct();

for idx_t = 1:numel(selected_trajs)
    tr_id = selected_trajs(idx_t);
    tr = trajs{tr_id};
    fprintf('>>> Simulando Escenario %d: %s ...\n', tr_id, tr.name);
    for c = 1:n_ctrl
        ctrl = controllers{c};
        ctrl.reset();
        t_N = numel(tr.t);
        ctrl_fn = @(tk, state_k, info_k) ctrl.compute(tk, state_k, info_k, tr.q_ref(:, min(max(1, info_k.k), t_N)));
        tic_id = tic();
        sim_out = stewart_simulate_plant(tr.t, ctrl_fn, [P.pose0; zeros(6,1)], tr.P_override, tr.tau_dist_fn);
        cpu_s = toc(tic_id);
        m_tab = calcular_metricas_tabla1(sim_out, tr.q_ref, tr.P_override, cpu_s);
        
        fprintf('    %-28s RMS pos %.3f mm | RMS ang %.3f deg | ITAE %.4g | Ju %.1f N^2s | Fmax %.1f N, Fsat %d, carrera %d | CPU %.3f ms/muestra\n', ...
            ctrl.name, m_tab.RMS_pos*1e3, rad2deg(m_tab.RMS_ang), m_tab.ITAE, m_tab.Ju, ...
            m_tab.Fmax, m_tab.force_sat_count, m_tab.stroke_sat_count, m_tab.cpu_time_per_sample_ms);
        
        results_e1.scen(idx_t).traj_id = tr_id;
        results_e1.scen(idx_t).traj_name = tr.name;
        results_e1.scen(idx_t).t = tr.t;
        results_e1.scen(idx_t).q_ref = tr.q_ref;
        results_e1.scen(idx_t).ctrl(c).name = ctrl.name;
        results_e1.scen(idx_t).ctrl(c).pose = sim_out.pose;
        results_e1.scen(idx_t).ctrl(c).twist = sim_out.twist;
        results_e1.scen(idx_t).ctrl(c).Fapplied = sim_out.Fapplied;
        results_e1.scen(idx_t).ctrl(c).L = sim_out.L;
        results_e1.scen(idx_t).ctrl(c).metrics = m_tab;
    end
end

out_file = fullfile(root_dir, '..', 'results', 'resultados_entrega1_octave.mat');
save('-v7', out_file, 'results_e1', 'sys_nom');
fprintf('\n[OK] Benchmark completado y guardado en: %s\n', out_file);
