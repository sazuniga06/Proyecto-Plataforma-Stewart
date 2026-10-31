function P = stewart_params()
%STEWART_PARAMS Parametros del gemelo Stewart 6-SPS en lazo abierto.
%
% Revision derivada de stewart_photo_twin_controlled.zip:
%   - se conservan la geometria corregida, IK/FK, Jacobiano y gemelo 3D;
%   - se eliminan PID y control cinematico realimentado;
%   - las patas son actuadores lineales cuya entrada es la fuerza axial
%     (modelo dinamico de Lopes, 2009);
%   - la referencia RPY se genera aleatoriamente, se sincroniza por IK y se
%     escala para respetar restricciones geometricas y diferenciales.

P.units = 'SI';

%% Geometria de anclajes (misma geometria corregida)
P.Rb = 0.170;          % [m] radio de anclajes base
P.Rp = 0.125;          % [m] radio de anclajes plataforma movil
P.baseAnglesDeg = [-15 15 105 135 225 255];
% Los seis agujeros superiores se conservan, pero se corrige el emparejamiento
% pata-base/pata-plataforma para que los actuadores no se crucen.
% Orden original de agujeros de la placa superior:
P.platHoleAnglesDeg = [45 315 165 75 285 195];
% Mapa mecanicamente admisible: B_i -> agujero superior topConnectionMap(i).
% Equivale a angulos [315 45 75 165 195 285] y elimina los cruces nominales.
P.topConnectionMap = [2 1 4 3 6 5];
P.platAnglesDeg = P.platHoleAnglesDeg(P.topConnectionMap);

ab = deg2rad(P.baseAnglesDeg(:));
ap = deg2rad(P.platAnglesDeg(:));
P.B = [P.Rb*cos(ab), P.Rb*sin(ab), zeros(6,1)];
P.A = [P.Rp*cos(ap), P.Rp*sin(ap), zeros(6,1)];

%% Pose nominal
P.h0 = 0.270;                         % [m]
P.pose0 = [0;0;P.h0;0;0;0];          % [x y z roll pitch yaw]

%% Geometria de los seis actuadores
P.stroke = 0.100;                     % [m] carrera total nominal
P.lowerBodyLength = 0.160;            % [m]
P.rodBodyLength   = 0.100;            % [m]
P.bodyRadius = 0.012;                 % [m]
P.rodRadius  = 0.006;                 % [m]

P.L0 = stewart_ik(P.pose0,P);
P.L0mean = mean(P.L0);
P.p0 = P.L0mean - P.lowerBodyLength - P.rodBodyLength;
P.pMin = max(0, P.p0 - P.stroke/2);
P.pMax = P.pMin + P.stroke;
P.Lmin = P.lowerBodyLength + P.rodBodyLength + P.pMin;
P.Lmax = P.lowerBodyLength + P.rodBodyLength + P.pMax;

%% Tiempo de simulacion
P.dt = 0.002;                         % [s] 500 Hz

%% Restricciones geometricas / holonomicas
% Cierre de cada pata:
%   Phi_i(q,L) = ||p + R(q) a_i - b_i|| - L_i = 0.
% La referencia mantiene el centro de la plataforma en [0 0 h0].
P.constraints.strokeMargin = 0.006;   % [m]
P.constraints.maxCondJ = 60;
P.constraints.maxAbsRPYdeg = [5 5 7]; % [roll pitch yaw]
P.constraints.maxHolonomicResidual = 1e-6; % [m]

%% Restricciones diferenciales (nivel velocidad)
% La compatibilidad de velocidad se evalua como
%   Psi = J(q)*xi - Ldot = 0,
% xi=[vx vy vz wx wy wz]^T, con omega en el marco base.
% Nota: esta ecuacion es la derivada del cierre holonomico y, por tanto,
% estrictamente es integrable; se conserva aqui como restriccion
% diferencial/Pfaffiana solicitada para la simulacion.
P.constraints.maxLegRate = 0.025;     % [m/s]
P.constraints.maxLegAccel = 0.080;    % [m/s^2]
P.constraints.maxAngularRateDeg = [12 12 16]; % [deg/s]
P.constraints.maxVelocityResidual = 5e-4; % [m/s], diagnostico numerico

%% Restricciones mecanicas de no penetracion y articulaciones
% Los actuadores se modelan como solidos tipo capsula (cuerpo + vastago).
% Se exige separacion positiva entre cualquier par de solidos de patas.
P.constraints.minLegSurfaceClearance = 0.005; % [m] holgura minima entre superficies
P.constraints.maxBaseJointTiltDeg = 32;       % [deg] limite editable de rotula inferior
P.constraints.maxTopJointTiltDeg  = 32;       % [deg] limite editable de rotula superior
P.constraints.enforceMechanicalProjection = true;
P.constraints.projectionIterations = 24;
P.constraints.projectionMinBeta = 1e-5;
P.constraints.stopVelocityOnProjection = true;

%% Excitacion RPY aleatoria independiente y sincronizada por IK
P.random.seed = 20260906;
P.random.nHarmonics = 8;
P.random.freqBandHz = [0.035 0.32];
P.random.maxRPYdeg = P.constraints.maxAbsRPYdeg;
P.random.rampTime = 2.0;              % [s]
P.random.minScale = 0.05;

%% Actuadores (modelo de Lopes, 2009): cada pata ejerce una fuerza axial F_i, que es la entrada de control
% de la planta: M(q) qdd + C(q,qd) qd + g(q) = (J Tq)' F + tau_ext. No se modela el accionamiento interno.
P.act.Fmax = 50.0;                    % [N] fuerza maxima por pata (supuesto de diseno, ~10 veces la estatica)

%% Plataforma movil para carga/inercia de feedforward
P.massTop = 2.0;                       % [kg]
P.massLower = 0.45;                    % [kg]
P.massRod = 0.12;                      % [kg]
P.gravity = 9.80665;                   % [m/s^2]

%% Cuerpos de cada pata (modelo de Lopes, 2009): cilindro unido a B_i y piston unido a P_i
% Cilindro y piston como cilindros macizos; centro de masa a la mitad de cada cuerpo.
P.leg.mC = P.massLower;                                 % [kg] masa del cilindro
P.leg.mS = P.massRod;                                   % [kg] masa del piston
P.leg.bC = P.lowerBodyLength/2;                         % [m] distancia B_i -> centro de masa del cilindro
P.leg.bS = P.rodBodyLength/2;                           % [m] distancia P_i -> centro de masa del piston
P.leg.IC = [0.5*P.leg.mC*P.bodyRadius^2, ...            % [kg*m^2] inercia axial del cilindro
            P.leg.mC*(3*P.bodyRadius^2 + P.lowerBodyLength^2)/12];   % inercia transversal
P.leg.IS = [0.5*P.leg.mS*P.rodRadius^2, ...
            P.leg.mS*(3*P.rodRadius^2 + P.rodBodyLength^2)/12];
%% Geometria visual (sin cambios)
P.baseHexRadius = 0.195;
P.topHexRadius  = 0.150;
P.baseBeamWidth = 0.020;
P.baseBeamHeight= 0.020;
P.topThickness  = 0.005;

% Inercia equivalente de la plataforma (aprox. disco fino, editable).
P.platformInertiaBody = diag([ ...
    0.25*P.massTop*P.topHexRadius^2, ...
    0.25*P.massTop*P.topHexRadius^2, ...
    0.50*P.massTop*P.topHexRadius^2]);

end
