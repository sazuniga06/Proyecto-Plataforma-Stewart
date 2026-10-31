function res = validar_modelo_dinamico(out_mat_path)
%VALIDAR_MODELO_DINAMICO Verificacion del modelo dinamico por momento generalizado (stewart_dynamic_matrices.m).
%   1. Propiedades de M(q): simetria y definicion positiva; aporte de la plataforma y de las patas.
%   2. Conservacion de la energia: sin gravedad ni fuerzas, la energia cinetica 1/2 qd' M qd
%      debe permanecer constante (prueba la consistencia del termino de Coriolis C).
%   3. Gravedad frente al gradiente numerico de la energia potencial de la plataforma, y dinamica
%      inversa F = (J T)^-T (M qdd + C qd + g) de la plataforma en la maniobra q0 -> q1 -> q0.
% Guarda los resultados en results/validacion_modelo.mat.

engine_dir = fullfile(fileparts(mfilename('fullpath')), '..');
addpath(fullfile(engine_dir, 'config'), fullfile(engine_dir, 'kinematics'), fullfile(engine_dir, 'dynamics'));
if nargin < 1 || isempty(out_mat_path)
    out_mat_path = fullfile(engine_dir, '..', 'results', 'validacion_modelo.mat');
end

% ---------------------------------------------------------------- 1. plataforma del proyecto
P = stewart_params();
q0 = P.pose0;
[M, ~, g, ~, J, T, info] = stewart_dynamic_matrices(q0, zeros(6,1), P);
res.M0 = M; res.g0 = g; res.info = info;
res.err_sim = norm(M - M.', 'fro');
res.eig_min = min(eig(M));
F0 = (J*T).' \ g;
res.F0 = F0;
printf('[1] Pose nominal: diag(M) = %s\n', mat2str(diag(M).', 6));
printf('    g_z = %.4f N; F0 = %.4f N por pata; min eig(M) = %.4f\n', g(3), F0(1), res.eig_min);

% ---------------------------------------------------------------- 2. conservacion de la energia
P2 = P; P2.gravity = 0;
x = [q0 + [0.01; -0.008; 0.01; 0.03; -0.02; 0.05]; [0.02; -0.03; 0.01; 0.2; -0.15; 0.3]];
f = @(x) [x(7:12); acel(x, P2)];
h = 1e-3; N = 500; E = zeros(N+1, 1);
E(1) = energia(x, P2);
for k = 1:N
    k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
    x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
    E(k+1) = energia(x, P2);
end
res.energia = E;
res.deriva_energia = max(abs(E - E(1))) / E(1);
printf('[2] Energia cinetica en 0.5 s de movimiento libre: deriva relativa maxima = %.2e\n', res.deriva_energia);

% ---------------------------------------------------------------- 3. gravedad y dinamica inversa de la plataforma
% fuerza estatica con g(q) y con el gradiente numerico de la energia potencial calculada solo con la geometria
gnum = zeros(6,1); hq = 1e-6;
for j = 1:6
    e = zeros(6,1); e(j) = hq;
    gnum(j) = (potencial(q0 + e, P) - potencial(q0 - e, P)) / (2*hq);
end
res.F_estatica = F0;
res.F_estatica_potencial = (J*T).' \ gnum;
printf('[3] Fuerza estatica: con g(q) = %.6f N; con el gradiente de la energia potencial = %.6f N\n', F0(1), res.F_estatica_potencial(1));

% maniobra q0 -> q1 -> q0 de la plataforma con perfiles trigonometricos: ida en 2 s, regreso en 3 s
q1 = q0 + [0.010; -0.008; 0.015; deg2rad(3); deg2rad(-3); deg2rad(4)];
dt = 0.01; t = (0:dt:7).';
q = zeros(6, numel(t)); qd = q; qdd = q; F = q; Lg = q;
for k = 1:numel(t)
    if t(k) <= 2
        [s, sd, sdd] = perfil_trig(t(k), 2); a = q0; b = q1;
    elseif t(k) <= 5
        [s, sd, sdd] = perfil_trig(t(k) - 2, 3); a = q1; b = q0;
    else
        s = 1; sd = 0; sdd = 0; a = q1; b = q0;
    end
    q(:,k) = a + s*(b - a); qd(:,k) = sd*(b - a); qdd(:,k) = sdd*(b - a);
    [Mk, Ck, gk, ~, Jk, Tk] = stewart_dynamic_matrices(q(:,k), qd(:,k), P);
    F(:,k) = (Jk*Tk).' \ (Mk*qdd(:,k) + Ck*qd(:,k) + gk);
    Lg(:,k) = stewart_ik(q(:,k), P);
end
res.man_t = t; res.man_F = F; res.man_L = Lg;
printf('    dinamica inversa de la maniobra: fuerzas entre %.3f y %.3f N\n', min(F(:)), max(F(:)));

save('-v7', out_mat_path, '-struct', 'res');
printf('Resultados guardados en %s\n', out_mat_path);
end

% -----------------------------------------------------------------------------------------------------
function a = acel(x, P)
[M, C, g] = stewart_dynamic_matrices(x(1:6), x(7:12), P);
a = M \ (-C*x(7:12) - g);
end

function E = energia(x, P)
M = stewart_dynamic_matrices(x(1:6), zeros(6,1), P);
E = 0.5 * x(7:12).' * M * x(7:12);
end

function V = potencial(q, L)
% Energia potencial: plataforma, cilindros (centro a bC de B_i) y pistones (centro a bS de P_i)
[l, n, Q] = stewart_ik(q, L);
zC = L.B(:,3) + L.leg.bC * n(:,3);
zS = Q(:,3) - L.leg.bS * n(:,3);
V = L.gravity * (L.massTop * q(3) + L.leg.mC * sum(zC) + L.leg.mS * sum(zS));
end

function [s, sd, sdd] = perfil_trig(t, T)
% Perfil trigonometrico (cicloidal) de reposo a reposo: s = tau - sin(2 pi tau)/(2 pi), con velocidad y
% aceleracion nulas en los extremos
tau = min(max(t / T, 0), 1);
s = tau - sin(2*pi*tau) / (2*pi);
sd = (1 - cos(2*pi*tau)) / T;
sdd = 2*pi*sin(2*pi*tau) / T^2;
end
