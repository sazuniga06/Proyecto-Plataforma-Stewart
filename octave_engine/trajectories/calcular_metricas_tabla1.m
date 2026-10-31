function metrics = calcular_metricas_tabla1(sim_out, q_ref, P, elapsed_cpu_total)
%CALCULAR_METRICAS_TABLA1 Computo exhaustivo de las metricas del protocolo comun (Tabla 1).
%
% Calcula metricas temporales, indices integrales, esfuerzo de control y costo computacional
% segun las Secciones 7 y 8 del documento del curso integrador.
%
% Entradas:
%   sim_out           - Salida de stewart_simulate_plant
%   q_ref             - Trayectoria de referencia 6xN
%   P                 - Parametros de la plataforma
%   elapsed_cpu_total - Tiempo de CPU total en segundos medido con tic/toc
%
% Salidas:
%   metrics - Estructura con todas las metricas escalares y formateadas.

t = sim_out.t;
N = numel(t);
Ts = P.dt;

pose_act = sim_out.pose;
e_pos = pose_act(1:3, :) - q_ref(1:3, :);
e_ang = pose_act(4:6, :) - q_ref(4:6, :);
e_all = pose_act - q_ref;

% 1. Metricas Temporales RMS
metrics.RMS_pos = sqrt(mean(sum(e_pos.^2, 1))); % [m]
metrics.RMS_ang = sqrt(mean(sum(e_ang.^2, 1))); % [rad]
metrics.RMS_pose = sqrt(mean(sum(e_all.^2, 1)));

% Error maximo
metrics.max_err_pos = max(sqrt(sum(e_pos.^2, 1)));
metrics.max_err_ang = max(sqrt(sum(e_ang.^2, 1)));

% Error estacionario en el ultimo 10% del tiempo
tail_idx = round(0.90 * N):N;
metrics.ss_error_pos = mean(sqrt(sum(e_pos(:, tail_idx).^2, 1)));
metrics.ss_error_ang = mean(sqrt(sum(e_ang(:, tail_idx).^2, 1)));

% Sobrepaso porcentual maximo en posicion
p_max = max(sqrt(sum(pose_act(1:3, :).^2, 1)));
p_ref_max = max(sqrt(sum(q_ref(1:3, :).^2, 1)));
if p_ref_max > 1e-4
    metrics.overshoot_pct = max(0, ((p_max - p_ref_max) / p_ref_max) * 100);
else
    metrics.overshoot_pct = 0;
end

% 2. Indices Integrales
% IAE = integral ||e(t)||_1 dt
norm_e1 = sum(abs(e_all), 1);
metrics.IAE = sum(norm_e1) * Ts;

% ISE = integral e(t)'*e(t) dt
norm_e2_sq = sum(e_all.^2, 1);
metrics.ISE = sum(norm_e2_sq) * Ts;

% ITAE = integral t * ||e(t)||_1 dt
metrics.ITAE = sum(t .* norm_e1) * Ts;

% 3. Esfuerzo de Control
% Ju = integral u(t)'*u(t) dt, con u = F las fuerzas axiales [N^2 s]
F_applied = sim_out.Fapplied;
metrics.Ju = sum(sum(F_applied.^2, 1)) * Ts;

metrics.Fmax = max(max(abs(F_applied)));
metrics.force_sat_count = nnz(sim_out.forceSat);
metrics.stroke_sat_count  = nnz(sim_out.strokeSat);

% 4. Margenes Mecanicos de Seguridad
metrics.min_clearance = min(sim_out.minClearance); % [m]
metrics.max_base_tilt = max(max(abs(sim_out.baseJointTilt))); % [deg]
metrics.max_top_tilt  = max(max(abs(sim_out.topJointTilt)));  % [deg]

% 5. Costo Computacional
if nargin >= 4 && ~isempty(elapsed_cpu_total)
    metrics.cpu_time_total_s = elapsed_cpu_total;
    metrics.cpu_time_per_sample_ms = (elapsed_cpu_total / max(1, N-1)) * 1000;
else
    metrics.cpu_time_total_s = 0;
    metrics.cpu_time_per_sample_ms = 0;
end
end
