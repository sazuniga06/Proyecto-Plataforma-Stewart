function id = identificacion_experimental(T, out_mat_path, sigma_L, solo_validar)
%IDENTIFICACION_EXPERIMENTAL Identificacion MIMO de las patas sobre la planta no lineal.
%
% 1. Ensayo (stewart_simulate_plant, planta no lineal): un control PD suave por pata mantiene la
%    plataforma en el espacio de trabajo y a la fuerza se suma la excitacion persistente de
%    generar_excitacion_prbs (multiseno + PRBS, independiente por pata):
%        F = F0 + Kp (L0 - L) - Kd (L - L_ant)/Ts + w   (velocidad por diferencia de la longitud medida)
%    Se registran la fuerza aplicada F (tras saturacion) y las longitudes MEDIDAS Lm = L + v, con ruido de
%    medicion gaussiano v de desviacion sigma_L (por defecto 0.05 mm); el PD del ensayo usa tambien Lm.
%    Se hacen dos ensayos independientes (distinta semilla de la PRBS y del ruido): el primero para estimar
%    y el segundo solo para validar (validacion cruzada).
% 2. Salida identificada: desviacion de longitud de cada pata, y(k) = L(k) - L0, con entrada dF = F - F0.
% 3. Modelo ARX(2,2) MIMO por minimos cuadrados (metodo directo en lazo cerrado). La pose tiene 12 estados y
%    se miden 6 salidas, de modo que el modelo discreto exacto es un ARX de orden 2:
%        y(k) = A1 y(k-1) + A2 y(k-2) + B0 dF(k) + B1 dF(k-1)
%    Se estima de dos formas con el primer ensayo:
%      a) directo, a Ts = 2 ms: con ruido en los regresores y desplazamientos por muestra del orden del ruido,
%         los minimos cuadrados quedan fuertemente sesgados;
%      b) con prefiltro y diezmado: Butterworth de orden 4 (fc = 20 Hz, fase cero, ida y vuelta) sobre y y dF y
%         diezmado por 5 (Ts_id = 10 ms), de modo que la senal por muestra crece frente al ruido.
%    Ambos se validan con el segundo ensayo por prediccion a horizonte fijo de 0.1 s (50 pasos a 2 ms, 10 pasos
%    a 10 ms): en cada instante el modelo parte de las mediciones y predice 0.1 s hacia adelante con la fuerza
%    medida. Como la planta es inestable en lazo abierto, la simulacion libre diverge y el FIT cae con el
%    horizonte para cualquier modelo; por eso se reporta tambien el FIT del modelo analitico con el mismo
%    procedimiento, como referencia de lo que es alcanzable.
% 4. Caja gris: del modelo analitico (Ad, Bd a Ts_id, salida J0 Cd) se deducen los coeficientes ARX exactos,
%    y(k+2) = C Ad^2 O^-1 [y(k); y(k+1)] + ..., O = [C; C Ad]. Se comparan la diagonal de A1 (2 cosh(sigma Ts_id)
%    para una pata aislada) y la masa efectiva m_i = Ts_id^2 / (B0 + B1)_ii.
%
% Entradas: T [s] duracion de cada ensayo (por defecto 20); out_mat_path (por defecto results/identificacion.mat);
%           sigma_L [m] ruido de medicion de las longitudes (por defecto 0.05e-3)
% Salidas en results/identificacion.mat: ys1/FIT1/VAF1 (ARX directo a 2 ms), ys2/FIT2/VAF2 (ARX prefiltrado) y
% FIT_an (modelo analitico). solo_validar = true recalcula la validacion con los ensayos y modelos ya guardados.

if nargin < 1 || isempty(T), T = 20; end
if nargin < 3 || isempty(sigma_L), sigma_L = 0.05e-3; end
if nargin < 4 || isempty(solo_validar), solo_validar = false; end
engine_dir = fullfile(fileparts(mfilename('fullpath')), '..');
if nargin < 2 || isempty(out_mat_path)
    out_mat_path = fullfile(engine_dir, '..', 'results', 'identificacion.mat');
end

P = stewart_params();
sys_nom = linealizacion_analitica('mechanical', P);
u0 = sys_nom.u0(:);
Ts = P.dt;
L0 = stewart_ik(P.pose0, P);

Kp_id = 1500; Kd_id = 60;          % Kp > lambda_max(K_l) = 428 N/m: el lazo del ensayo es estable
D = 5; Ts_id = D * Ts; fc = 20;                    % diezmado y frecuencia de corte del prefiltro
[bf, af] = butter4(fc, 1 / Ts);
if solo_validar
    S = load(out_mat_path);
    t = S.t; y = S.y; dF = S.dF; w = S.w; L = S.L; y_e = S.y_est; Th1 = S.Th1; Th2 = S.Th2; sigma_L = S.sigma_L;
    dF_e = []; if isfield(S, 'dF_est'), dF_e = S.dF_est; end
else
    [t, y_e, dF_e] = ensayo(T, 20260919, 1);       % ensayo de estimacion
    [~, y, dF, w, L] = ensayo(T, 20261017, 2);     % ensayo de validacion (independiente)
    % a) ARX(2,2) directo a 2 ms
    Th1 = arx22(y_e, dF_e);
    % b) ARX(2,2) con prefiltro de fase cero y diezmado (Ts_id = 10 ms)
    Th2 = arx22(dec(filtfilt0(bf, af, y_e), D), dec(filtfilt0(bf, af, dF_e), D));
end

    function [t, y, dF, w, L] = ensayo(T, semilla, semilla_ruido)
        [w, t] = generar_excitacion_prbs(T, Ts, P, zeros(6, 1), semilla);
        randn('state', semilla_ruido);
        v = sigma_L * randn(6, numel(t));            % ruido de medicion de las longitudes
        Lm_ant = L0;
        function u = ctrl(~, ~, info)
            Lm = info.L + v(:, info.k);
            u = u0 + Kp_id * (L0 - Lm) - Kd_id * (Lm - Lm_ant) / Ts + w(:, info.k);
            Lm_ant = Lm;
        end
        sim_out = stewart_simulate_plant(t, @ctrl, [P.pose0; zeros(6,1)], P);
        % Datos: salida medida y(k) y la fuerza aplicada en el mismo intervalo (F(:,k) actua sobre (t_{k-1}, t_k])
        L = sim_out.L + v;
        y = L - L0;
        dF = sim_out.Fapplied - u0;
        dF(:, 1) = 0;
    end

N = numel(t);
N_est = N;
% Caja gris: coeficientes ARX exactos del modelo analitico a Ts_id
[M0, ~, ~, ~, J0, T0] = stewart_dynamic_matrices(P.pose0, zeros(6,1), P);
Jq = J0 * T0;
nA = size(sys_nom.A, 1);
E = expm([sys_nom.A, sys_nom.B; zeros(6, nA + 6)] * Ts_id);
C = Jq * sys_nom.Cd; Ad = E(1:nA, 1:nA); Bd = E(1:nA, nA+1:end);
O = [C; C * Ad];
G = C * Ad^2 / O;                          % [A2_an, A1_an]
A2_an = G(:, 1:6); A1_an = G(:, 7:12);
B0_an = C * Bd;                            % coeficiente de dF(k) (u actuando en el ultimo intervalo)
B1_an = C * Ad * Bd - G(:, 7:12) * C * Bd; % coeficiente de dF(k-1)
a_an = diag(A1_an);
m_an = Ts_id^2 ./ diag(B0_an + B1_an);
A1 = Th2(:, 1:6); A2 = Th2(:, 7:12); B0 = Th2(:, 13:18); B1 = Th2(:, 19:24);
a_id = diag(A1);
m_id = Ts_id^2 ./ diag(B0 + B1);
m_id_directo = Ts^2 ./ diag(Th1(:, 13:18) + Th1(:, 19:24));

% Validacion con el segundo ensayo: prediccion a 0.1 s
h = 0.1;
ys1 = predecir(Th1, y, dF, round(h / Ts));
y2 = dec(filtfilt0(bf, af, y), D); u2 = dec(filtfilt0(bf, af, dF), D);
ys2 = predecir(Th2, y2, u2, round(h / Ts_id));
ysa = predecir([A1_an, A2_an, B0_an, B1_an], y2, u2, round(h / Ts_id));
val = find(all(isfinite(ys1), 1));
v2 = find(all(isfinite(ys2), 1));
m1 = fit_metrics(y(:, val), ys1(:, val));
m2 = fit_metrics(y2(:, v2), ys2(:, v2));
ma = fit_metrics(y2(:, v2), ysa(:, v2));

% Respuesta en frecuencia dF_1 -> l_1 a Ts_id: modelo analitico frente al ARX prefiltrado
wf = logspace(-1, log10(0.95 * pi / Ts_id), 300);
z = exp(1i * wf * Ts_id);
G_an = zeros(size(wf)); G_id = G_an;
for kk = 1:numel(wf)
    Gm = C * ((z(kk) * eye(nA) - Ad) \ Bd);
    G_an(kk) = Gm(1, 1);
    zi = 1 / z(kk);
    Gi = (eye(6) - A1 * zi - A2 * zi^2) \ (B0 + B1 * zi);
    G_id(kk) = Gi(1, 1);
end

id = struct('t', t, 'y', y, 'dF', dF, 'L', L, 'val_idx', val, 'ys1', ys1, 'ys2', ys2, ...
            'VAF1', m1.VAF, 'FIT1', m1.FIT, 'RMSE1', m1.RMSE, 'VAF2', m2.VAF, 'FIT2', m2.FIT, 'RMSE2', m2.RMSE, ...
            'Th1', Th1, 'Th2', Th2, 'a_id', a_id, 'm_id', m_id, 'a_an', a_an, 'm_an', m_an, ...
            'wf', wf, 'G_an', G_an, 'G_id', G_id, 'Kp_id', Kp_id, 'Kd_id', Kd_id, 'N_est', N_est, 'Ts', Ts, ...
            'sigma_L', sigma_L, 'y_est', y_e, 'w', w, 'Ts_id', Ts_id, 'fc_id', fc, 'D_id', D, ...
            'm_id_directo', m_id_directo, 'dF_est', dF_e, 't2', t(1:D:end), 'y2', y2, 'v2', v2, 'h_val', h, ...
            'FIT_an', ma.FIT, 'VAF_an', ma.VAF, 'ysa', ysa);
save('-v7', out_mat_path, '-struct', 'id');

fprintf('\n=== Identificacion experimental (planta no lineal, 2 ensayos de %d muestras, Ts = %.3f s, ruido %.3f mm) ===\n', N, Ts, 1e3 * sigma_L);
fprintf('ARX directo (2 ms):       VAF medio = %.2f %%, FIT medio = %.2f %%, RMSE medio = %.4f mm, m_i = [%s] kg\n', mean(m1.VAF), mean(m1.FIT), 1e3 * mean(m1.RMSE), sprintf('%.3f ', m_id_directo));
fprintf('ARX prefiltrado (10 ms):  VAF medio = %.2f %%, FIT medio = %.2f %%, RMSE medio = %.4f mm\n', mean(m2.VAF), mean(m2.FIT), 1e3 * mean(m2.RMSE));
fprintf('Modelo analitico (10 ms): VAF medio = %.2f %%, FIT medio = %.2f %%  (prediccion a %.1f s)\n', mean(ma.VAF), mean(ma.FIT), h);
fprintf('Caja gris identificada: a_ii = [%s], m_i = [%s] kg\n', sprintf('%.5f ', a_id), sprintf('%.3f ', m_id));
fprintf('Modelo analitico:       a_ii = [%s], m_i = [%s] kg\n', sprintf('%.5f ', a_an), sprintf('%.3f ', m_an));
fprintf('Resultados guardados en: %s\n', out_mat_path);
end

function Th = arx22(y, u)
% ARX(2,2) MIMO por minimos cuadrados: y(k) = A1 y(k-1) + A2 y(k-2) + B0 u(k) + B1 u(k-1)
k = 3:size(y, 2);
Th = y(:, k) / [y(:, k-1); y(:, k-2); u(:, k); u(:, k-1)];
end

function x = dec(x, D)
x = x(:, 1:D:end);
end

function yp = predecir(Th, y, u, h)
% Prediccion a h pasos del ARX(2,2): en cada instante k0 parte de las mediciones y(k0-1), y(k0-2) y propaga h
% pasos con la fuerza medida; yp(:, k0+h-1) es la prediccion de y(k0+h-1) (NaN donde no hay prediccion).
A1 = Th(:, 1:6); A2 = Th(:, 7:12); B0 = Th(:, 13:18); B1 = Th(:, 19:24);
n = size(y, 2);
yp = nan(size(y));
for k0 = 3:n-h+1
    ym1 = y(:, k0-1); ym2 = y(:, k0-2);
    for j = 0:h-1
        k = k0 + j;
        yk = A1 * ym1 + A2 * ym2 + B0 * u(:, k) + B1 * u(:, k-1);
        ym2 = ym1; ym1 = yk;
    end
    yp(:, k0+h-1) = yk;
end
end

function [b, a] = butter4(fc, fs)
% Butterworth pasa bajos de orden 4 por transformacion bilineal con predistorsion (sin el paquete signal)
n = 4;
wc = 2 * fs * tan(pi * fc / fs);
pa = wc * exp(1i * pi * (2 * (1:n) + n - 1) / (2 * n));
pz = (1 + pa / (2 * fs)) ./ (1 - pa / (2 * fs));
a = real(poly(pz));
b = poly(-ones(1, n));
b = b * sum(a) / sum(b);                           % ganancia unitaria en continua
end

function yf = filtfilt0(b, a, y)
% Filtrado de fase cero: hacia adelante y hacia atras sobre cada fila
yf = filter(b, a, y, [], 2);
yf = fliplr(filter(b, a, fliplr(yf), [], 2));
end

function m = fit_metrics(y, ys)
e = y - ys;
m.VAF = 100 * (1 - var(e, 0, 2) ./ var(y, 0, 2));
m.FIT = 100 * (1 - sqrt(sum(e.^2, 2)) ./ sqrt(sum((y - mean(y, 2)).^2, 2)));
m.RMSE = sqrt(mean(e.^2, 2));
end
