function ctrl = controlador_estado_observador(P, sys_nom, mode, opts)
%CONTROLADOR_ESTADO_OBSERVADOR Estrategia 2: Realimentacion de estados con observador de Luenberger.
%
% Fundamentado en el Modulo 05 del Cerebro Principal de Obsidian:
%   - 05.3: Asignacion modal de polos u = -K*x (eig(A - B*K) = P_d(s))
%   - 05.4: Controlabilidad de Kalman rank([B, AB, ..., A^(n-1)B]) = n
%   - 05.5: Precompensador N_r = -[C*(A - B*K)^(-1)*B]^(-1) (y verificacion por Cofactores/Adjunta)
%   - 05.6: Accion Integral en Espacio de Estados (Controlador PI de Estado x_a = [x; x_i])
%   - 05.7: Observador de Luenberger por Dualidad de Kalman (A <-> A^T, B <-> C^T, K <-> L^T)
%   - 05.8: Principio de Separacion en la matriz bloque-triangular A_total (24x24)

base_dir = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(base_dir, 'config'));
addpath(fullfile(base_dir, 'kinematics'));
addpath(fullfile(base_dir, 'dynamics'));
addpath(fullfile(base_dir, 'solvers'));

if nargin < 1 || isempty(P), P = stewart_params(); end
if nargin < 2 || isempty(sys_nom)
    sys_nom = linealizacion_analitica('mechanical', P);
end
if nargin < 3 || isempty(mode), mode = 'precompensador'; end
if nargin < 4 || isempty(opts), opts = struct(); end

pole_scale = 1.0;
obs_scale  = 1.0;
if isfield(opts, 'pole_speed_factor'), pole_scale = opts.pole_speed_factor; end
if isfield(opts, 'obs_speed_factor'),  obs_scale  = opts.obs_speed_factor;  end

A = sys_nom.A;
B = sys_nom.B;
C = sys_nom.C;
Ad = sys_nom.Ad;
Bd = sys_nom.Bd;
Cd = sys_nom.Cd;
u0 = sys_nom.u0;
x0_nom = sys_nom.x0;
y0_nom = x0_nom(1:6);
n = sys_nom.n;
m = sys_nom.m;
p = sys_nom.p;
Ts = sys_nom.Ts;

% 1. Asignacion de polos de control (Modulo 05.3 / 05.6)
ctrl_poles = pole_scale * 3 * [-8.5+3.0i, -8.5-3.0i, ...
                               -8.8+3.2i, -8.8-3.2i, ...
                               -9.1+3.4i, -9.1-3.4i, ...
                               -9.4+3.6i, -9.4-3.6i, ...
                               -9.7+3.8i, -9.7-3.8i, ...
                               -10.0+4.0i, -10.0-4.0i];
Ki = zeros(m, p);
if strcmpi(mode, 'integral_pi')
    A_aug = [A, zeros(n, p); -C, zeros(p, p)];
    B_aug = [B; zeros(p, m)];
    int_poles = pole_scale * 3 * [-1.5, -1.6, -1.7, -1.8, -1.9, -2.0];
    aug_poles = [ctrl_poles, int_poles];
    [K_aug, ~, poles_cl] = octave_place_mimo(A_aug, B_aug, aug_poles);
    K  = K_aug(:, 1:n);
    Ki = -K_aug(:, n+1:end); % Convencion Nota 05.6: K_a = [K, -k_i]
else
    [K, ~, poles_cl] = octave_place_mimo(A, B, ctrl_poles);
end

% 3. Precompensador de referencia Nr (Modulo 05.5 y Tarea 5)
A_cl = A - B * K;
DC_gain = C * (A_cl \ B);
if rcond(DC_gain) > 1e-10
    Nr = -inv(DC_gain);
else
    Nr = -pinv(DC_gain);
end

% Verificacion analitica por Matriz Adjunta y Cofactores (Tarea 5 / Nota 05.5):
det_Acl = det(A_cl);
Adj_Acl = det_Acl * inv(A_cl);
DC_gain_adj = (C * Adj_Acl * B) / det_Acl;
Nr_cofactores = -inv(DC_gain_adj);
err_Nr_cof = norm(Nr - Nr_cofactores, 'fro') / max(1, norm(Nr, 'fro'));

% 4. Diseno del Observador de Luenberger por Dualidad de Kalman (Modulo 05.7)
obs_poles = obs_scale * 3 * [-26.0+6.0i, -26.0-6.0i, ...
                             -27.5+6.5i, -27.5-6.5i, ...
                             -29.0+7.0i, -29.0-7.0i, ...
                             -30.5+7.5i, -30.5-7.5i, ...
                             -32.0+8.0i, -32.0-8.0i, ...
                             -33.5+8.5i, -33.5-8.5i];
[Lt, ~, poles_obs] = octave_place_mimo(A.', C.', obs_poles);
L_cont = Lt.';

% Ganancia del observador discreto en forma de ESTIMADOR ACTUAL (Franklin-Powell):
%   correccion  xc(k)   = xp(k) + Ld (y(k) - Cd xp(k))
%   prediccion  xp(k+1) = Ad xc(k) + Bd u(k)
% El error de prediccion ep = x - xp evoluciona con ep(k+1) = Ad (I - Ld Cd) ep(k), semejante a
% (I - Ld Cd) Ad = Ad - Ld (Cd Ad). Por dualidad, Ld se obtiene asignando los polos de (Ad', (Cd Ad)').
obs_poles_disc = exp(obs_poles * Ts);
[Ldt, ~, ~] = octave_place_mimo(Ad.', (Cd * Ad).', obs_poles_disc);
Ld = Ldt.';
obs_poles_impl = eig(Ad * (eye(n) - Ld * Cd));   % polos realmente implementados (verificacion)

% 5. Verificacion del Principio de Separacion (Modulo 05.8)
A_total = [A - B * K, B * K; zeros(n, n), A - L_cont * C];
eig_total = sort(real(eig(A_total)));
eig_union = sort(real([eig(A - B * K); eig(A - L_cont * C)]));
sep_error = norm(eig_total - eig_union);

state_mem = struct();
state_mem.x_hat = zeros(n, 1);
state_mem.x_hat_corr = zeros(n, 1);
state_mem.x_int = zeros(p, 1);

    function u = compute_sf_obs(t, state, info, q_ref)
        if nargin < 4 || isempty(q_ref)
            q_ref = P.pose0;
        end
        y_curr = state(1:6);
        delta_y = y_curr - y0_nom;
        delta_r = q_ref - y0_nom;
        err_track = q_ref - y_curr;
        
        obs_error = delta_y - Cd * state_mem.x_hat;
        x_hat_corr = state_mem.x_hat + Ld * obs_error;
        state_mem.x_hat_corr = x_hat_corr;
        
        if strcmpi(mode, 'integral_pi')
            delta_u = -K * x_hat_corr + Nr * delta_r + Ki * state_mem.x_int;
        else
            delta_u = -K * x_hat_corr + Nr * delta_r;
        end
        
        u_raw = u0 + delta_u;
        u = max(min(u_raw, P.act.Fmax), -P.act.Fmax);
        
        if strcmpi(mode, 'integral_pi')
            if all(abs(u_raw) <= P.act.Fmax)
                state_mem.x_int = max(min(state_mem.x_int + Ts * err_track, 0.05), -0.05);
            end
        end
        
        u_applied = u - u0;
        state_mem.x_hat = Ad * x_hat_corr + Bd * u_applied;
    end

    function reset_sf_obs(x_hat0)
        if nargin >= 1 && ~isempty(x_hat0)
            state_mem.x_hat = x_hat0(:);
            state_mem.x_hat_corr = x_hat0(:);
        else
            state_mem.x_hat = zeros(n, 1);
            state_mem.x_hat_corr = zeros(n, 1);
        end
        state_mem.x_int = zeros(p, 1);
    end

    function xh = get_x_hat()
        xh = x0_nom + state_mem.x_hat_corr;
    end

ctrl.compute = @compute_sf_obs;
ctrl.reset   = @reset_sf_obs;
ctrl.get_x_hat = @get_x_hat;
if strcmpi(mode, 'integral_pi')
    ctrl.name = 'Realim. Estados PI + Observador';
else
    ctrl.name = 'Realim. Estados (Nr) + Observador';
end
ctrl.mode = mode;
ctrl.K = K;
ctrl.Ki = Ki;
ctrl.L = L_cont;
ctrl.Ld = Ld;
ctrl.Nr = Nr;
ctrl.Nr_cofactores = Nr_cofactores;
ctrl.err_Nr_cof = err_Nr_cof;
ctrl.ctrl_poles = ctrl_poles;
ctrl.obs_poles = obs_poles;
ctrl.poles_cl = poles_cl;
ctrl.poles_obs = poles_obs;
ctrl.obs_poles_disc = obs_poles_disc;
ctrl.obs_poles_impl = obs_poles_impl;
ctrl.A_total = A_total;
ctrl.sep_error = sep_error;
end
