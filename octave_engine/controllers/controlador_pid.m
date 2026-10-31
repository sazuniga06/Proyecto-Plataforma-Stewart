function ctrl = controlador_pid(P, sys_nom, mode, opts)
%CONTROLADOR_PID PID descentralizado en el espacio de las patas con anti-windup.
%
%   e_i = L_ref,i - L_i,  F_i = F0_i + Kp e_i + Ki x_I,i + Kd d_i   (fuerza axial de la pata i, N)
%   x_I,i(k+1) = x_I,i(k) + Ts e_i(k)   (integracion condicional: se congela si satura en el mismo sentido)
%   d_i(k) = a d_i(k-1) + (1 - a) (e_i(k) - e_i(k-1)) / Ts,  a = exp(-wf Ts), wf = 160 rad/s (derivada filtrada)
%
% Entradas:
%   P       - Parametros de la plataforma (stewart_params)
%   sys_nom - Sistema linealizado nominal (linealizacion_analitica); u0 = F0
%   mode    - se conserva por compatibilidad ('leg_space')
%   opts    - Estructura opcional con escaladores Kp_scale, Ki_scale, Kd_scale

base_dir = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(base_dir, 'config'));
addpath(fullfile(base_dir, 'kinematics'));
addpath(fullfile(base_dir, 'dynamics'));

if nargin < 1 || isempty(P), P = stewart_params(); end
if nargin < 2 || isempty(sys_nom), sys_nom = linealizacion_analitica('mechanical', P); end
if nargin < 4 || isempty(opts), opts = struct(); end

kp_s = 1.0; ki_s = 1.0; kd_s = 1.0;
if isfield(opts, 'Kp_scale'), kp_s = opts.Kp_scale; end
if isfield(opts, 'Ki_scale'), ki_s = opts.Ki_scale; end
if isfield(opts, 'Kd_scale'), kd_s = opts.Kd_scale; end

u0 = sys_nom.u0;
Ts = P.dt;

% Ganancias de diseno por pata: Kp [N/m] supera la rigidez geometrica negativa de las patas (428 N/m),
% Kd [N s/m] amortigua los modos de la plataforma y Ki [N/(m s)] anula el error ante cargas constantes
Kp = kp_s * 6000.0 * ones(6, 1);
Ki = ki_s * 6000.0 * ones(6, 1);
Kd = kd_s *  150.0 * ones(6, 1);
af = exp(-160 * Ts);                     % polo del filtro derivativo

state_mem = struct('int_err', zeros(6, 1), 'prev_err', zeros(6, 1), 'd_filter', zeros(6, 1));

    function u = compute_pid(~, ~, info, q_ref)
        if nargin < 4 || isempty(q_ref), q_ref = P.pose0; end
        err = stewart_ik(q_ref, P) - info.L;
        state_mem.d_filter = af * state_mem.d_filter + (1 - af) * (err - state_mem.prev_err) / Ts;
        u_raw = u0 + Kp .* err + Ki .* state_mem.int_err + Kd .* state_mem.d_filter;
        u = max(min(u_raw, P.act.Fmax), -P.act.Fmax);
        not_saturated = (u_raw == u) | (sign(err) ~= sign(u_raw));
        state_mem.int_err = state_mem.int_err + Ts * (err .* not_saturated);
        state_mem.prev_err = err;
    end

    function reset_pid()
        state_mem = struct('int_err', zeros(6, 1), 'prev_err', zeros(6, 1), 'd_filter', zeros(6, 1));
    end

ctrl.compute = @compute_pid;
ctrl.reset   = @reset_pid;
ctrl.name    = 'PID (leg_space)';
ctrl.Kp = Kp;
ctrl.Ki = Ki;
ctrl.Kd = Kd;
ctrl.alpha_d = af;
end
