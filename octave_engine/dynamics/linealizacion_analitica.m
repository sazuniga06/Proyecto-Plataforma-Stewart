function sys = linealizacion_analitica(state_mode, P)
%LINEALIZACION_ANALITICA Obtencion del modelo linealizado en espacio de estados y verificacion de Kalman.
%
% Obtiene las matrices continuas (A, B, C, D) y discretas (Ad, Bd, Cd, Dd)
% de la plataforma Stewart alrededor de la pose nominal de reposo q0 = [0 0 h0 0 0 0]'.
%
% Modelo de Lopes con la fuerza axial de cada pata como entrada: x = [q; qdot] (n = 12), u = F (m = 6).
% En el equilibrio (qdot = 0, C = 0) la linealizacion analitica es
%   A = [0, I; -M0^-1 dG/dq, 0],  B = [0; M0^-1 (J0 T0)'],  G(q) = g(q) - (J T)' F0
% y aqui se verifica por diferencias finitas centrales sobre f(x, u).
%
% Entradas:
%   state_mode - se conserva por compatibilidad (solo existe el modelo mecanico de 12 estados)
%   P          - Parametros de la planta (opcional, stewart_params)

base_dir = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(base_dir, 'config'));
addpath(fullfile(base_dir, 'kinematics'));
addpath(fullfile(base_dir, 'dynamics'));

if nargin < 2 || isempty(P)
    P = stewart_params();
end
q0 = P.pose0;
qdot0 = zeros(6,1);

% 1. Punto de operacion: fuerzas axiales que sostienen el peso, (J0 T0)' F0 = g0
[~, ~, g0, ~, J0, Tq0] = stewart_dynamic_matrices(q0, qdot0, P);
Jq0 = J0 * Tq0;
Fleg0 = Jq0.' \ g0;

n = 12; m = 6; p = 6;
x0 = [q0; qdot0];
u0 = Fleg0;

% 2. Calculo numerico de derivadas parciales por diferencias finitas centrales
%    A = df/dx |_(x0, u0),   B = df/du |_(x0, u0)
A = zeros(n, n);
B = zeros(n, m);

hx = 1e-6;
for j = 1:n
    x_plus = x0;  x_plus(j) = x_plus(j) + hx;
    x_minus = x0; x_minus(j) = x_minus(j) - hx;
    f_plus  = stewart_plant_state_derivative(x_plus, u0, zeros(6,1), P);
    f_minus = stewart_plant_state_derivative(x_minus, u0, zeros(6,1), P);
    A(:, j) = (f_plus - f_minus) / (2 * hx);
end

hu = 1e-5;
for j = 1:m
    u_plus = u0;  u_plus(j) = u_plus(j) + hu;
    u_minus = u0; u_minus(j) = u_minus(j) - hu;
    f_plus  = stewart_plant_state_derivative(x0, u_plus, zeros(6,1), P);
    f_minus = stewart_plant_state_derivative(x0, u_minus, zeros(6,1), P);
    B(:, j) = (f_plus - f_minus) / (2 * hu);
end

% Matriz de salida C: medicion directa de la pose q (primeros 6 estados)
C = [eye(6), zeros(6, n - 6)];
D = zeros(p, m);

% 3. Discretizacion exacta por exponente de matriz de Van Loan
Ts = P.dt;
M_block = [A, B; zeros(m, n), zeros(m, m)];
Phi = expm(M_block * Ts);
Ad = Phi(1:n, 1:n);
Bd = Phi(1:n, n+1:n+m);
Cd = C;
Dd = D;

% 4. Comprobacion de Controlabilidad y Observabilidad (Kalman)
% Matriz de controlabilidad: Mc = [B, A*B, ..., A^(n-1)*B]
Mc = zeros(n, n * m);
col = B;
for k = 1:n
    Mc(:, (k-1)*m + 1 : k*m) = col;
    col = A * col;
end
rank_Mc = rank(Mc, 1e-7);
is_controllable = (rank_Mc == n);

% Matriz de observabilidad: Mo = [C; C*A; ...; C*A^(n-1)]
Mo = zeros(n * p, n);
row = C;
for k = 1:n
    Mo((k-1)*p + 1 : k*p, :) = row;
    row = row * A;
end
rank_Mo = rank(Mo, 1e-7);
is_observable = (rank_Mo == n);

% 5. Polos en lazo abierto
poles_cont = eig(A);
poles_disc = eig(Ad);

% Guardar resultados en estructura
sys.n = n;
sys.m = m;
sys.p = p;
sys.x0 = x0;
sys.u0 = u0;
sys.Fleg0 = Fleg0;
sys.Ts = Ts;
sys.A = A;
sys.B = B;
sys.C = C;
sys.D = D;
sys.Ad = Ad;
sys.Bd = Bd;
sys.Cd = Cd;
sys.Dd = Dd;
sys.Mc = Mc;
sys.Mo = Mo;
sys.rank_Mc = rank_Mc;
sys.rank_Mo = rank_Mo;
sys.is_controllable = is_controllable;
sys.is_observable = is_observable;
sys.poles_cont = poles_cont;
sys.poles_disc = poles_disc;

fprintf('=== Linealizacion Plataforma Stewart (n = %d) ===\n', n);
fprintf('Punto de operacion: z0 = %.3f m, F0 promedio = %.3f N\n', q0(3), mean(Fleg0));
fprintf('Rango de Controlabilidad (Kalman): %d / %d -> %s\n', rank_Mc, n, tf_str(is_controllable));
fprintf('Rango de Observabilidad (Kalman):   %d / %d -> %s\n', rank_Mo, n, tf_str(is_observable));
end

function s = tf_str(b)
if b, s = 'SI (Completamente Controlable/Observable)'; else, s = 'NO'; end
end
