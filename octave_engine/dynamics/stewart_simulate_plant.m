function sim_out = stewart_simulate_plant(t_span, controller_fn, x0, P, tau_dist_fn)
%STEWART_SIMULATE_PLANT Simulacion no lineal de la plataforma Stewart 6-SPS (modelo de Lopes, 2009).
%
% Integra con RK4 (un paso por muestra, Ts = 2 ms) la dinamica completa
%   M(q) qddot + C(q, qdot) qdot + g(q) = (J Tq)' F + tau_d(t)
% con la fuerza axial de cada pata F como entrada, saturada a +-P.act.Fmax y mantenida constante
% durante cada periodo de muestreo (retenedor de orden cero). En cada muestra se registran ademas las
% longitudes de las patas y, cada 10 muestras, los margenes mecanicos (holgura entre patas y giro de las
% rotulas; en las muestras intermedias se repite el ultimo valor).
%
% Entradas:
%   t_span        - Vector de tiempo [0:Ts:Tend] o escalar Tend
%   controller_fn - u = controller_fn(t_k, x_k, info_k) (info_k.L, .Ldot, .k, .Ts, .pose, .twist)
%                   o matriz de fuerzas 6xN en lazo abierto
%   x0            - Estado inicial [q0; qdot0] (opcional)
%   P             - Parametros (stewart_params)
%   tau_dist_fn   - Perturbacion generalizada tau_d(t) 6x1 (opcional)
%
% Salida: sim_out con .t, .pose, .qdot, .twist, .L, .Ldot, .Fapplied, .minClearance, .baseJointTilt,
%         .topJointTilt, .forceSat, .strokeSat

if nargin < 4 || isempty(P), P = stewart_params(); end
if isscalar(t_span), t = 0:P.dt:t_span; else, t = t_span(:).'; end
if nargin < 3 || isempty(x0), x0 = [P.pose0; zeros(6,1)]; end
if nargin < 5 || isempty(tau_dist_fn), tau_dist_fn = @(tk) zeros(6,1); end
N = numel(t); Ts = P.dt; nsub = 1; h = Ts / nsub;
open_loop = isnumeric(controller_fn);

X = zeros(12, N); X(:, 1) = x0(:);
twist = zeros(6, N); L = zeros(6, N); Ldot = zeros(6, N); F = zeros(6, N);
min_clear = zeros(1, N); base_tilt = zeros(6, N); top_tilt = zeros(6, N);
force_sat = false(6, N); stroke_sat = false(6, N);

for k = 1:N
    x = X(:, k);
    [~, ~, ~, ~, J, Tq] = stewart_dynamic_matrices(x(1:6), zeros(6,1), P);
    twist(:, k) = Tq * x(7:12);
    L(:, k) = stewart_ik(x(1:6), P);
    Ldot(:, k) = J * twist(:, k);
    if mod(k - 1, 10) == 0
        Mk = stewart_mechanical_constraints(x(1:6), P);
    end
    min_clear(k) = Mk.minSurfaceClearance;
    base_tilt(:, k) = Mk.baseJointTiltDeg;
    top_tilt(:, k) = Mk.topJointTiltDeg;
    stroke_sat(:, k) = L(:, k) < P.Lmin | L(:, k) > P.Lmax;
    if k == N, break; end

    if open_loop
        u = controller_fn(:, k);
    else
        info = struct('k', k, 'Ts', Ts, 'L', L(:, k), 'Ldot', Ldot(:, k), 'pose', x(1:6), 'twist', twist(:, k));
        u = controller_fn(t(k), x, info);
    end
    u = u(:);
    force_sat(:, k + 1) = abs(u) > P.act.Fmax;
    F(:, k + 1) = max(min(u, P.act.Fmax), -P.act.Fmax);

    tau = tau_dist_fn(t(k));
    f = @(xx) stewart_plant_state_derivative(xx, F(:, k + 1), tau, P);
    for s = 1:nsub
        k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
        x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
    end
    X(:, k + 1) = x;
end

sim_out = struct('t', t, 'pose', X(1:6, :), 'qdot', X(7:12, :), 'twist', twist, 'L', L, 'Ldot', Ldot, ...
                 'Fapplied', F, 'minClearance', min_clear, 'baseJointTilt', base_tilt, 'topJointTilt', top_tilt, ...
                 'forceSat', force_sat, 'strokeSat', stroke_sat);
end
