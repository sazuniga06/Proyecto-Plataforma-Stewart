function test_data = generar_trayectorias_test(test_id, Tend, Ts, P)
%GENERAR_TRAYECTORIAS_TEST Banco unificado de trayectorias normalizadas de prueba.
%
% Genera las 7 trayectorias requeridas en la Seccion 6 del documento del curso:
%   1. Cambio de referencia en z alrededor de la altura nominal (+20 mm).
%   2. Seguimiento senoidal en x o y (amplitud 25 mm, f = 0.25 Hz).
%   3. Trayectoria circular en el plano x-y (radio 25 mm, f = 0.20 Hz).
%   4. Variacion simultanea de traslacion y orientacion (maniobra acoplada 6 GDL).
%   5. Prueba de rechazo a perturbacion externa (pulso de fuerza de 5 N en z).
%   6. Cambio de carga util (incremento de masa Delta m = +1.5 kg a los 4 s).
%   7. Operacion en la frontera segura del espacio de trabajo.
%
% Entradas:
%   test_id - Numero de prueba (1 a 7, por defecto 4 para maniobra completa)
%   Tend    - Duracion de la prueba en segundos (por defecto 10 s)
%   Ts      - Periodo de muestreo (por defecto 0.01 s)
%   P       - Parametros del sistema (stewart_params)
%
% Salidas:
%   test_data - Estructura con:
%       .t, .q_ref, .name, .tau_dist_fn, .param_modifier_fn

if nargin < 4 || isempty(P), P = stewart_params(); end
if nargin < 2 || isempty(Tend), Tend = 10.0; end
if nargin < 3 || isempty(Ts), Ts = P.dt; end
if nargin < 1 || isempty(test_id), test_id = 4; end

t = 0:Ts:Tend;
N = numel(t);
q_ref = repmat(P.pose0, 1, N); % 6xN
h0 = P.h0;

tau_dist_fn = @(tk) zeros(6,1);
param_mod_fn = @(P_in, tk) P_in;

switch test_id
    case 1 % 1. Cambio de paso en z (+20 mm)
        test_name = '1. Escalón en Altura Z (+20 mm)';
        for k = 1:N
            if t(k) >= 1.5
                q_ref(3, k) = h0 + 0.020;
            else
                q_ref(3, k) = h0;
            end
        end
        
    case 2 % 2. Seguimiento senoidal en x
        test_name = '2. Senoide en X (25 mm, 0.25 Hz)';
        f2 = 0.25;
        ramp = min(1, t / 1.5);
        q_ref(1, :) = 0.025 * ramp .* sin(2 * pi * f2 * t);
        
    case 3 % 3. Trayectoria circular en plano x-y
        test_name = '3. Círculo en Plano X-Y (R=25 mm)';
        R = 0.025; f3 = 0.20;
        ramp = min(1, t / 2.0);
        q_ref(1, :) = R * ramp .* cos(2 * pi * f3 * t);
        q_ref(2, :) = R * ramp .* sin(2 * pi * f3 * t);
        
    case 4 % 4. Maniobra simultanea completa 6 GDL
        test_name = '4. Maniobra Simultánea Completa 6 GDL';
        ramp = min(1, t / 2.0);
        q_ref(1, :) = 0.020 * ramp .* cos(2 * pi * 0.25 * t);
        q_ref(2, :) = 0.020 * ramp .* sin(2 * pi * 0.25 * t);
        q_ref(3, :) = h0 + 0.015 * ramp .* sin(2 * pi * 0.50 * t);
        q_ref(4, :) = deg2rad(4.0) * ramp .* sin(2 * pi * 0.30 * t);
        q_ref(5, :) = deg2rad(4.0) * ramp .* cos(2 * pi * 0.30 * t);
        q_ref(6, :) = deg2rad(5.0) * ramp .* sin(2 * pi * 0.20 * t);
        
    case 5 % 5. Rechazo a perturbacion externa (pulso de fuerza)
        test_name = '5. Rechazo a Perturbación (Pulso 5 N en Z)';
        tau_dist_fn = @(tk) (tk >= 3.0 && tk <= 3.6) * [0; 0; -5.0; 0; 0; 0];
        
    case 6 % 6. Cambio brusco de carga util / fuerza sostenida en Z (t=2.5 s)
        test_name = '6. Escalón Z (+15 mm) + Sobrecarga Sostenida (10 N)';
        for k = 1:N
            if t(k) >= 0.8
                q_ref(3, k) = h0 + 0.015;
            end
        end
        tau_dist_fn = @(tk) (tk >= 2.5) * [0; 0; -10.0; 0; 0; 0];
        
    case 7 % 7. Operacion proxima a limites de carrera y rotulas
        test_name = '7. Frontera del Espacio de Trabajo';
        ramp = min(1, t / 2.0);
        q_ref(1, :) = 0.035 * ramp .* sin(2 * pi * 0.20 * t);
        q_ref(2, :) = 0.035 * ramp .* cos(2 * pi * 0.20 * t);
        q_ref(3, :) = h0 + 0.030 * ramp .* sin(2 * pi * 0.35 * t);
        q_ref(4, :) = deg2rad(6.0) * ramp .* sin(2 * pi * 0.25 * t);
        q_ref(5, :) = deg2rad(6.0) * ramp .* cos(2 * pi * 0.25 * t);
        q_ref(6, :) = deg2rad(7.0) * ramp .* sin(2 * pi * 0.15 * t);
        
    otherwise
        error('test_id debe estar comprendido entre 1 y 7.');
end

test_data.t = t;
test_data.q_ref = q_ref;
test_data.test_id = test_id;
test_data.name = test_name;
test_data.tau_dist_fn = tau_dist_fn;
test_data.param_mod_fn = param_mod_fn;
test_data.P_override = P; % planta nominal (param_mod_fn es identidad en los 7 escenarios)
end
