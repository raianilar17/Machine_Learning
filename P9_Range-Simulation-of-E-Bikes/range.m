function range()
  
  V = [0;3.5;7;10.6;14.1;17.6;21.5;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;
  25;25;25;25;25;25;25;25;25;25;25;25;25;25;25;21.6;17.6;14;10.6;7;3.5;0];
  
  N = length(V);
  
  % Assume
  printf('Total Time %d Second of Datasets \n\n',N)
  printf('velocity in KM/hr \n\n')
  V'
  
  % now velocity in km/hr , so convert into m/sec(divide by 3.6) 
  printf('length of velocity  %d \n\n',N)
  
  
  V = V./3.6;
  printf('velocity in m/sec \n\n' )
  V'
  
  %printf('\n\n');
  
  % initialize vechile values
  
  vech_mass = 25 ; % vechile_mass in kg
  passengers_mass = 75 ; % total_passengers_mass in  kg
  m = 100;  % total_mass_of_body in kg
  g = 9.81 ; % acceleration_of_gravity in m/sec^2
  A = 0.4 ; % frontal_area in m^2
  Cd = 0.7 ; % Drag_Coefficient
  Gratio = 1/2; % Gearing ratio = G/r
  G_eff = 0.90 ; % Transmission efficiency    
  Regen_ratio  = 0.90 ; % This sets the proportion of braking that is done regeneratively using the motor
  mu_rr = 0.0063 ; % rolling resistance coefficient
  rho = 1.125 ; % density _of_air kg/m^3
  psi = 0; % Angle_of_hill_climb in degree
  r = 0.25 ; % Radius_of_metre in m
  I  = 0.45 ; % Moment_of_inertia_rotor_of_motorin Kg.m^2
  dT = 1; % DeltaT(change in time)
  
  bat_type = "li-ion" ; % Lithium_ion_battery
  NoCells = 40 ; % 10 of 4 cell (36v) Battery
  Capacity = 9.6 ; % 9.6 Ah batteries . This is assumed to be the 10 hour rate capacity
  k = 1.09 ; % peukert coefficient
  Pac = 50 ; % Average power of accessories
  E = 36; %volt
  
  % constants for motor efficiency
  %Kc = 0.3 ; % For copper losses coefficient
  %Ki = 0.01 ; % For iron losses coefficient
  %Kw = 0.000005 ; % For windage losses coefficient
  %ConL = 600 ; % For constant electronics losses
  
  Kc = 1.05; % For copper losses coefficient
  Ki = 0.1; % For iron losses coefficient
  Kw = 0.00001 ; % For windage losses coefficient
  ConL = 20; % For constant electronics losses
  
  % some constants which are calculated
  Rin = (0.022/Capacity)*NoCells;
  Rin = Rin + 0.05; % add a little internal resistance to allow for connecting cables
  PeuCap = ((Capacity/10)^k) *10;
  %PeuCap = 1;
  
  % Set up the arrays for storing data for battery ,and distance travelld. ALL set to zero at start.
  % These first arrays are for storing the value at the end of each cycle.
  % Intially, assume that no more than 100 of any cycle is completed.(if there is any error then adjust these values).
  DoD_end = zeros(1,100);
  CR_end = zeros(1,100);
  D_end = zeros(1,100);
  
  DoD = zeros(1,N); % depth of Discharge
  CR = zeros(1,N);  % charge removed from the battery ,Peukert Corrected
  D = zeros(1,N);  % Record of distance travelled in km
  
 CY = 1; % CY control the outer loop, counts the number of cycles completed . we want to keep cycling till the battery is flat.
 % This we define as being more than 90% discharged. this is DoD_end > 0.9
 % use the variable XX to monitor the discharge, and to stop the loop going too far.
         
  DD = 0 ; % initially zero  
  
  while DD < 0.9
    %Beginning of cycle!
    %printf('Cycle number is %d \n', CY )
    for C = 2:N
       accel = (V(C)-V(C-1))/dT;
       %printf(" C value is %d and acceleration is %f \n",C,accel);
    
       Frr = mu_rr * m * g;
       Fad = (1/2) * rho * A * Cd * (V(C)^2);
       psi_radians = psi * (pi/180);
       Fhc = m * g * sin(psi_radians);
       Fla = m*accel;
       %Fla = 0;
       Fwa = I *(1/G_eff)*((Gratio)^2)*accel;
    
       Fte = (Frr + Fad + Fhc + Fla + Fwa);
    
       %printf(" C value is %d and tractive effort is %f \n",C,Fte);
    
       Pte = Fte * V(C);  % energy required to move the vechile for each sec
       %printf(" C value is %d and power of tractive effort is %f \n",C,Pte);
    
       omega = Gratio * V(C); % omega is a motor angular speed in rad/sec
       %printf(" C value is %d and angular speed is %f \n",C,omega);
    
       if omega ==0 % stationary
           Pte = 0;
           Pmot_in = 0; % NO Power in motor
           Torque = 0;
           eff_mot = 0.5 ; % Dummy Value, to make sure not zeros
       elseif omega > 0; % moving
           if Pte < 0
               Pte = Regen_ratio * Pte; % Reduce the power if braking , as not all will be by the motor
            end;
        
          % calculate the output power of motor , Which is different from that at the wheels , because of transmissions losses.
          if Pte >= 0
            Pmot_out = Pte / G_eff; % motor power > shaft power
          elseif Pte < 0;
            Pmot_out = Pte * G_eff; % motor power diminished if engine braking
          end;
       
        
            Torque = Pmot_out / omega; % Basic equation  P = T * omega;
            %printf(" C value is %d and Torque is %f \n",C,Torque);
         
         
           if Torque >=0
              eff_mot = ((Torque*omega)/((Torque*omega)+ (Kc*(Torque^2)) + (Ki*omega) +(Kw*(omega^3)) + ConL));
           elseif Torque <0
             eff_mot = ((-Torque*omega)/((-Torque*omega)+ (Kc*(Torque^2)) + (Ki*omega) +(Kw*(omega^3)) + ConL));
           end;
          
            %printf(" C value is %d and motor efficiency is %f \n",C,eff_mot);
            %printf(" C value is %d and motor output is %f \n",C,Pmot_out);
            eff_mot = 0.80; %usually 15%
         
           if Pmot_out > 0
              Pmot_in = Pmot_out / eff_mot;
           elseif Pmot_out < 0
            Pmot_in = Pmot_out * eff_mot;
          end;
        end;
        
        %printf(" C value is %d and Pmot_in is %f \n",C,Pmot_in);
        Pbat = Pmot_in + Pac;
        %printf(" C value is %d and Pbat is %f \n",C,Pbat);
        
        %E = Open_Circuit_voltage(0, NoCells);
        
        if Pbat > 0
          I = (E - ((E*E) - (4*Rin*Pbat))^0.5)/(2*Rin);
          CR(C) = CR(C-1) + (((dT)*(I^k))/3600);
        elseif Pbat == 0
          I = 0;
        elseif Pbat<0
          % Regenerative Braking double the internal resistance
          Pbat = -1 * Pbat;
          I = (-E + ((E*E) + (4*Rin*Pbat))^0.5)/(2*Rin);
          CR(C) = CR(C-1) - (((dT)*(I))/3600);
        end;
        
        %printf(" C value is %d and I is %f \n",C, I);
        %printf(" C value is %d and CR is %f \n",C, CR(C));
        DoD(C) = CR(C)/PeuCap;
        %printf(" C value is %d and DoD is %f \n",C, DoD);
        
        if DoD(C)>1
          DoD(C) = 1;
        end;
        
        % Since we are taking one second time intervals , the distance traveled in metres is the same as velocity . Divide by 1000 for km.
        D(C) = D(C-1) + (V(C)/1000);
        %D(C) = D(C-1) + (V(C)*3.6);
        XDATA(C) = C;
        YDATA(C) = eff_mot;
    end;
    
     % one complete cycle done 
     % Now update the end of cycle values.
     DoD_end(CY) = DoD(N);
     CR_end(CY) = CR(N);
     D_end(CY) = D(N);
     
     % Now reset the values of these "inner" arrays ready for the next cycle.
     % They should start where they left off
     
     DoD(1) = DoD(N);
     CR(1) = CR(N);
     D(1) = D(N);
     DD = DoD_end(CY); % Update state of discharge.
     %printf('depth of discharge %d in CY is %d \n',DD,CY);
     
     % End of cycle!
     CY = CY + 1;
   end;
  printf('End of Depth of Discharge is %f\n',DD) 
  printf('Total Cycle Number is %d \n', CY )
  Range = D(N)*0.9/DoD(N);
  printf("Range is %d KMs\n", Range);
  
  %figure();
  %plot(XDATA, YDATA,'k+');
  %ylabel('Efficieny of motor');
  %xlabel('Cycle_number');
  
  figure();
  plot(D_end, DoD_end, 'k+');
  
  ylabel('Depth of Discharge');
  xlabel('Distance travelled /KM');
  %axis([0,10,0,1])
end

  
   
                   
