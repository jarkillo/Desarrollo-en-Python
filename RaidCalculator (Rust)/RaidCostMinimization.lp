\* RaidCostMinimization *\
Minimize
OBJ: 2200 num_C4 + 50 num_Explosive_5.56_Rifle_Ammo + 1400 num_Rocket
 + 480 num_Satchel_Charge
Subject To
_C1: 250 num_C4 + 4 num_Explosive_5.56_Rifle_Ammo + 220 num_Rocket
 + 70 num_Satchel_Charge >= 250
_C2: 250 num_C4 + 5.2 num_Explosive_5.56_Rifle_Ammo + 247 num_Rocket
 + 91.5 num_Satchel_Charge >= 250
Bounds
 0 <= num_C4
 0 <= num_Explosive_5.56_Rifle_Ammo
 0 <= num_Rocket
 0 <= num_Satchel_Charge
Generals
num_C4
num_Explosive_5.56_Rifle_Ammo
num_Rocket
num_Satchel_Charge
End
