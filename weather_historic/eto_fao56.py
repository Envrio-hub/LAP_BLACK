# The FAO 56 formula to estimate reference evapotranspiration
import numpy as np

def eto_fao56(data, constants, ts = "daily", solar = "sunshine hours", wind = "yes", crop = "short", message = "yes"):

    # Check for missing data
    if "Tmax" not in data.columns or "Tmin" not in data.columns:
        print("Required data missing for 'Tmax' and 'Tmin', or 'Temp'")
        exit()
    if "va" not in data.columns or "vs" not in data.columns:
        if "RHmax" not in data.columns or "RHmin" not in data.columns:
            print("Required data missing: need either 'va' and 'vs', or 'RHmax' and 'RHmin' (or 'RH')")
            exit()
    if wind == "yes":
        if "u2" not in data.columns and "uz" not in data.columns:
            print("Required data missing for 'uz' or 'u2'")
            exit()
    if solar == "data" and "Rs" not in data.columns and solar == "data" and "Rn" not in data.columns:
        print("Required data missing for 'Rs' or 'Rn'")
        exit()
    elif solar == "sunshine hours" and "n" not in data.columns:
        print("Required data missing for 'n'")
    elif solar == "cloud" and "Cd" not in data.columns:
        print("Required data missing for 'Cd'")
        exit()
    elif solar == "monthly precipitation" and "Precip" not in data.columns:
        print("Required data missing for 'Precip'")
    if wind != "yes" and wind != "no":
        print("Please choose if actual data will be used for wind speed from wind = 'yes' and wind = 'no'")
        exit()
    if wind == "yes":
        if crop != "short" and crop != "tall":
            print("Please enter 'short' or 'tall' for the desired reference crop type")
        else:
            alpha = 0.23
            if crop == "short":
                z0 = 0.02
            else:
                z0 = 0.1
    else:
        z0 = 0.02
        alpha = 0.25
    
    # Initita the calculation
    Ta = (data.Tmax + data.Tmin)/2
    if "va" in data.columns and "vs" in data.columns:
        vabar = data.va
        vas = data.vs
    else:
        vs_Tmax = 0.6108 * np.exp(17.27 * data.Tmax/(data.Tmax + 237.3))
        vs_Tmin = 0.6108 * np.exp(17.27 * data.Tmin/(data.Tmin + 237.3))
        vas = (vs_Tmax + vs_Tmin)/2
        vabar = (vs_Tmin * data.RHmax/100 + vs_Tmax * data.RHmin/100)/2
    P = 101.3 * ((293 - 0.0065 * constants["Elev"])/293)**5.26
    delta = 4098 * (0.6108 * np.exp((17.27 * Ta)/(Ta + 237.3)))/((Ta + 237.3)**2)
    gamma = 0.00163 * P/constants["lambda"]
    d_r2 = 1 + 0.033 * np.cos(2 * np.pi/365 * data.J)
    delta2 = 0.409 * np.sin(2 * np.pi/365 * data.J - 1.39)
    w_s = np.arccos(-np.tan(constants["lat_rad"] * np.tan(delta2)))
    N = 24/np.pi * w_s
    R_a = (1440/np.pi) * d_r2 * constants["Gsc"] * (w_s * np.sin(constants["lat_rad"] * 
           np.sin(delta2) + np.cos(constants["lat_rad"] * np.cos(delta2) * np.sin(w_s))))
    R_so = (0.75 + (2 * 10*-5) * constants["Elev"] * R_a)
    if solar == "data" and "Rn" in data.columns:
        R_ng = data.Rn
    else:
        if solar == "data" and "Rs" in data.columns:
            R_s = data.Rs
        elif solar == "data" and "Rn" in data.columns:
            R_ng = data.Rn
        elif solar != "monthly precipitation":
            R_s = (constants["as"] + constants["bs"] * (data.n/N))
        else:
            R_s = (0.85 - 0.047 * data.Cd) * R_a
        R_nl = constants["sigma"] * (0.34 - 0.14 * np.sqrt(vabar)) * ((data.Tmax + 273.16)*4 + (data.Tmin + 273.16)*4)/2 * (1.35 * R_s/R_so - 0.35)
        R_nsg = (1 - alpha) * R_s
        R_ng = R_nsg - R_nl
    if wind == "yes":
        if data.u2.empty:
            u2 = data.uz * 4.87/np.log(67.8 * constants["z"] - 5.42)
        else:
            u2 = data.u2
    if crop == "short":
        r_s = 70
        CH = 0.12
        ET_RC_Daily = (0.408 * delta *(R_ng - constants["G"]) + gamma * 900 * u2 * (vas - vabar)/(Ta +273))/(delta + gamma * (1+ 0.34 *u2))
    else:
        r_s = 45
        CH = 0.5
        ET_RC_Daily = (0.408 * delta * (R_ng - constants["G"]) + gamma * 1600 * u2 *(vas - vabar)/(Ta +273))/(delta + gamma * (1 + 0.38 *u2))
    ET_Daily = ET_RC_Daily

    # Prepare report data
    if wind == "no":
        ET_formulation = "Penman-Monteith (without wind data)",
        ET_type = "Reference Crop ET"
        Surface = f"short grass, albedo = {alpha}; roughness height ={z0} m"
    else:
        ET_formulation = "Penman-Monteith FAO56"
        ET_type = "Reference Crop ET"
        Surface = f"FAO-56 hypothetical short grass, albedo = {alpha}; surface resistance = {r_s} sm^1; crop height = {CH} m; roughness height = {z0} m"
    
    if solar == "data":
        message1 = "Solar radiation data have been used directly for calculating evapotranspiration"
    elif solar == "sunshine hours":
        message1 = "Sunshine hour data have been used for calculating incoming solar radiation"
    elif solar == "cloud":
        message1 = "Cloudiness data have been used for calculating sunshine hour and thus incoming solar radiation"
    else:
        message1 = "Monthly precipitation data have been used for calculating incoming solar radiation"

    if wind == "yes":
        message2 = "Wind data have been used for calculating the reference crop evapotranspiration"
    else:
        message2 = "Alternative calculation for reference crop evapotranspiration without wind data have been performed"
    
    if ts == "daily":
        resutls = {
            "ET_Daily": ET_Daily,
            "ET_formulation": ET_formulation,
            "ET_type": ET_type,
            "message1": message1,
            "message2": message2
            }
    return(resutls)
