import os
import shutil
import pandas as pd

def copy_files_by_name(root_folder, target_folder, filenames):
    foldername = r'D:\USER FILES\DESKTOP\WSU\Spring 2024\Neural Network\Project\Dataset\Index_dataset\full_history'
    for filename in filenames:
        source_file = os.path.join(root_folder, filename)
        if os.path.exists(source_file):
            if filename.endswith('.csv'):
                filepath = os.path.join(foldername, filename)
                try:
                    # Read CSV skipping first row as header
                    df = pd.read_csv(filepath)#, skiprows=1)
                    if not df.empty and len(df) >= 4000:
                        start_date = pd.to_datetime(df['date'].iloc[-1])
                        if start_date <= pd.Timestamp('2006-01-01'):
                            filtered_files.append(filename)
                            shutil.copy(filepath, os.path.join(target_folder, filename))

                    else:
                        print(f"Skipping file {filename} because it contains no data.")
                except KeyError:
                    print(f"Skipping file {filename} because 'date' column not found.")
        else:
            print(f"File '{filename}' not found in the root folder.")

root_folder = r'D:\USER FILES\DESKTOP\WSU\Spring 2024\Neural Network\Project\Dataset\Index_dataset\full_history'
target_folder = r'D:\USER FILES\DESKTOP\WSU\Spring 2024\Neural Network\Project\Dataset\Index_dataset\filtered_dataset\Finance'

# List of filenames to copy
filtered_files = ['ABCB.csv','AEL.csv','AFB.csv','AGD.csv','AGO.csv','AIZ.csv','AMSF.csv','AOD.csv','ARCC.csv','ATLC.csv','AVK.csv','AWF.csv','AWP.csv','AXS.csv','BANF.csv','BCBP.csv','BCV.csv','BDJ.csv','BFIN.csv','BFZ.csv','BGR.csv','BGT.csv','BGY.csv','BHB.csv','BHK.csv','BHV.csv','BKN.csv','BLE.csv','BLK.csv','BLX.csv','BME.csv','BOE.csv','BTA.csv','BTO.csv','BTZ.csv','BX.csv','BYM.csv','CACC.csv','CAF.csv','CASH.csv','CET.csv','CEV.csv','CGO.csv','CHCI.csv','CHI.csv','CHW.csv','CHY.csv','CIA.csv','CII.csv','CLM.csv','CME.csv','CMU.csv','CNS.csv','COF.csv','COLB.csv','CPSS.csv','CRESY.csv','CRF.csv','CSQ.csv','CXE.csv','CXH.csv','CZWI.csv','DGICB.csv','DHF.csv','DHY.csv','DMF.csv','DNP.csv','DSM.csv','DSU.csv','DTF.csv','EAD.csv','ECF.csv','EDD.csv','EFR.csv','EFT.csv','EGF.csv','EHI.csv','EHTH.csv','EIG.csv','EIM.csv','EMF.csv','ENX.csv','EOD.csv','EOI.csv','EOS.csv','ERC.csv','ERH.csv','ESSA.csv','ETB.csv','ETG.csv','ETJ.csv','ETO.csv','ETV.csv','ETW.csv','ETY.csv','EVF.csv','EVG.csv','EVM.csv','EVN.csv','EVR.csv','EVT.csv','EVV.csv','EXG.csv','FAM.csv','FANH.csv','FAX.csv','FBNC.csv','FCO.csv','FCT.csv','FEN.csv','FFA.csv','FFC.csv','FFNW.csv','FGB.csv','FISI.csv','FLC.csv','FMN.csv','FMY.csv','FOF.csv','FRA.csv','FRST.csv','FSFG.csv','FTF.csv','GAB.csv','GAIN.csv','GDL.csv','GDV.csv','GF.csv','GGN.csv','GLO.csv','GLQ.csv','GLRE.csv','GLU.csv','GLV.csv','GNW.csv','GS.csv','GSBC.csv','HBCP.csv','HDB.csv','HIX.csv','HMN.csv','HNW.csv','HOMB.csv','HPF.csv','HPI.csv','HPS.csv','HQH.csv','HQL.csv','HTD.csv','HYB.csv','HYT.csv','IAE.csv','IAF.csv','ICE.csv','IGA.csv','IGD.csv','IGR.csv','IHT.csv','IIF.csv','ING.csv','JCE.csv','JFR.csv','JOF.csv','JPC.csv','JQC.csv','JRS.csv','KFFB.csv','KRNY.csv','KSM.csv','KTF.csv','KYN.csv','LANV.csv','LAZ.csv','LEO.csv','LSBK.csv','MAIN.csv','MBI.csv','MBWM.csv','MCR.csv','MET.csv','MFD.csv','MFIC.csv','MFM.csv','MGF.csv','MGYR.csv','MHD.csv','MHF.csv','MHI.csv','MHLD.csv','MHN.csv','MIN.csv','MIY.csv','MKL.csv','MKTX.csv','MLP.csv','MMT.csv','MMU.csv','MORN.csv','MPA.csv','MPV.csv','MQT.csv','MQY.csv','MSD.csv','MTG.csv','MUA.csv','MUC.csv','MUI.csv','MUJ.csv','MVF.csv','MVT.csv','MYD.csv','MYN.csv','NAC.csv','NAD.csv','NAN.csv','NAZ.csv','NBH.csv','NCV.csv','NCZ.csv','NEA.csv','NFBK.csv','NFJ.csv','NIE.csv','NIM.csv','NKX.csv','NMI.csv','NMZ.csv','NNI.csv','NNY.csv','NOM.csv','NPV.csv','NQP.csv','NRK.csv','NRO.csv','NUV.csv','NVG.csv','NXC.csv','NXJ.csv','NXN.csv','NXP.csv','NZF.csv','OXSQ.csv','PCK.csv','PCM.csv','PCN.csv','PCQ.csv','PFD.csv','PFG.csv','PFL.csv','PFN.csv','PFO.csv','PFS.csv','PGP.csv','PHD.csv','PHK.csv','PIM.csv','PMF.csv','PML.csv','PMM.csv','PMO.csv','PMX.csv','PNF.csv','PNI.csv','PPT.csv','PRAA.csv','PRK.csv','PRU.csv','PSEC.csv','PTMN.csv','PTY.csv','PYN.csv','PZC.csv','RBCAA.csv','RCS.csv','RFI.csv','RNP.csv','RQI.csv','RVT.csv','SAFT.csv','SBI.csv','SCD.csv','SEIC.csv','SSBI.csv','TBBK.csv','TCBI.csv','TEI.csv','TFSL.csv','TMP.csv','TRC.csv','TROW.csv','TWN.csv','TYG.csv','USA.csv','UTF.csv','UTG.csv','VALU.csv','VFL.csv','VKI.csv','VLT.csv','VVR.csv','WAL.csv','WEA.csv','WIA.csv','WIW.csv','WRLD.csv','WSBF.csv','WTW.csv','ZTR.csv',]  # Replace with your list of filenames

copy_files_by_name(root_folder, target_folder, filtered_files)
