function [u_exp, t_exp] = generar_excitacion_prbs(Tend, Ts, P, u0, semilla)
%GENERAR_EXCITACION_PRBS Generacion de senales persistentes y seguras de excitacion MIMO.
%
% Genera senales compuestas por multiseno ortogonal y PRBS modulada para los 6 actuadores,
% garantizando que las fuerzas permanezcan dentro de los limites seguros y la plataforma
% no exceda su espacio de trabajo mecanicamente admisible.
%
% Entradas:
%   Tend - Tiempo total del ensayo (s), por defecto 20 s
%   Ts   - Periodo de muestreo (s), por defecto P.dt (0.01 s)
%   P    - Parametros de la plataforma
%   u0   - Punto de operacion en fuerza (6x1) [N]
%   semilla - semilla de la PRBS (por defecto 20260919); otra semilla da un ensayo independiente
%
% Salidas:
%   u_exp - Matriz de fuerzas de excitacion 6xN (N)
%   t_exp - Vector de tiempo 1xN (s)

if nargin < 1 || isempty(Tend), Tend = 20.0; end
if nargin < 3 || isempty(P), P = stewart_params(); end
if nargin < 2 || isempty(Ts), Ts = P.dt; end
if nargin < 4 || isempty(u0)
    sys_temp = linealizacion_analitica('mechanical', P);
    u0 = sys_temp.u0;
end

t_exp = 0:Ts:Tend;
N = numel(t_exp);
u_exp = zeros(6, N);

% Amplitud de la excitacion alrededor de u0 [N]
amp_multisine = 3.0;  % N
amp_prbs = 2.0;       % N

% Frecuencias fundamentales de excitacion para caracterizar la dinamica MIMO (0.1 Hz a 4.0 Hz)
freqs = [0.15, 0.35, 0.70, 1.20, 2.10, 3.50];

% Semilla determinista para reproducibilidad
if nargin < 5 || isempty(semilla), semilla = 20260919; end
rand('state', semilla);

for i = 1:6
    % Componente 1: Multiseno con fases de Schroeder para minimizar el factor de cresta
    fi = freqs(i);
    phase = (i * (i - 1) * pi) / 6;
    s_sine = amp_multisine * sin(2 * pi * fi * t_exp + phase) + ...
             0.5 * amp_multisine * sin(2 * pi * (fi * 1.7) * t_exp + phase/2);
    
    % Componente 2: PRBS binaria filtrada (cambio de estado cada 0.25 s)
    samples_per_bit = round(0.25 / Ts);
    num_bits = ceil(N / samples_per_bit);
    bit_seq = 2 * (rand(1, num_bits) > 0.5) - 1;
    s_prbs = repelem(bit_seq, samples_per_bit);
    s_prbs = amp_prbs * s_prbs(1:N);
    
    % Suavizado de transiciones con rampa inicial de 1.5 s
    ramp = min(1, t_exp / 1.5);
    u_raw = u0(i) + ramp .* (s_sine + s_prbs);
    
    % Limites de saturacion seguros
    u_exp(i, :) = max(min(u_raw, 0.85 * P.act.Fmax), -0.85 * P.act.Fmax);
end
end
