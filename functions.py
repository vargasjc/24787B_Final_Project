# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:07:44 2018

@author: varga
"""

import pyedflib
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

folder = r"D:\Projects\MLIA Set\CAP Sleep"

###########################################################
#
# These functions is to read edf files and store them as pkls
#
###########################################################

def process_file(record_loc):
    f = pyedflib.EdfReader(record_loc)
    file_dict = {
        #"FileInfoLong":f.file_info_long(),
        "FileDuration":f.getFileDuration(),
        "Header":f.getHeader(),
        "SignalHeaders":{x: f.getSignalHeader(chn=x) for x in range(f.signals_in_file)},
        "Data":{x: f.readSignal(chn=x,start=0,n=None) for x in range(f.signals_in_file)}
    }
    f._close()
    return file_dict

def process_files(folder):
    data = {}
    errors = []
    for file in os.listdir(folder):
        filename, extension = os.path.splitext(file)
        if extension == ".edf":
            record_loc = os.path.join(folder,file)
            try:
                print("\n========================\nProcessing:",file)
                file_dict = process_file(record_loc)
                data[filename] = file_dict
            except:
                print("Error processing file:",file)
                errors.append(file)
    return data, errors

def process_files_and_pickle(folder):
    errors = []
    for file in os.listdir(folder):
        filename, extension = os.path.splitext(file)
        if extension == ".edf":
            if not os.path.isfile(os.path.join(folder,"pkl",filename+".pkl")):
                record_loc = os.path.join(folder,file)
                try:
                    print("\n========================\nProcessing:",file)
                    file_dict = process_file(record_loc)
                    pickle.dump(file_dict,open(os.path.join(folder,"pkl",filename+".pkl"),"ab"))
                except:
                    print("Error processing file:",file)
                    errors.append(file)
            #else:
            #    print("\n========================\nFile already processed:",file)
    return errors

def load_file(record_loc, try_convertion = True):
    folder = os.path.dirname(record_loc)
    filename, extension = os.path.splitext(record_loc)
    filename = filename.replace(folder+"\\","")
    if extension == ".edf":
        pkl_file = os.path.join(folder,"pkl",filename+".pkl")
        if os.path.isfile(pkl_file):
            return pickle.load(open(pkl_file,"rb"))
        else:
            if try_convertion:
                print("Pickle not found! Attempting to parse .edf file.")
                return process_file(record_loc)
            else:
                print("Pickle not found! Skipping .edf file.")
    elif extension == ".pkl":
        return pickle.load(open(record_loc,"rb"))
    else:
        print("Unrecognized file:",record_loc)
        
#errors = process_files_and_pickle(folder)

###########################################################
#
# These functions are for data access and visualization
#
###########################################################

def plot_signal(record_dict, channel,filename = "", normalize=True):
    if filename!="":
        prefix = filename + ", channel=" + str(channel) + " -> "
    else:
        prefix = "channel=" + str(channel) + " -> "
    if normalize:
        gain = (record_dict["SignalHeaders"][channel]["physical_max"]-record_dict["SignalHeaders"][channel]["physical_min"])/(record_dict["SignalHeaders"][channel]["digital_max"]-record_dict["SignalHeaders"][channel]["digital_min"])
        data = record_dict["Data"][channel] * gain
    else:
        data = record_dict["Data"][channel]
    plt.figure(figsize=(15,5))
    plt.title(prefix + record_dict["SignalHeaders"][channel]["transducer"] + " - " + record_dict["SignalHeaders"][channel]["prefilter"])
    plt.plot(data, label=record_dict["SignalHeaders"][channel]["transducer"])
    plt.xlabel("Time Series->Samples (@"+str(record_dict["SignalHeaders"][channel]["sample_rate"])+"Hz)")
    plt.ylabel(record_dict["SignalHeaders"][channel]["label"] + " (" + record_dict["SignalHeaders"][channel]["dimension"] + ")    range: [" + str(record_dict["SignalHeaders"][channel]["physical_min"]) + "," + str(record_dict["SignalHeaders"][channel]["physical_max"]) + "]")
    plt.show()
    
def plot_file_signals(folder,filename, channel = None):
    record_loc = os.path.join(folder,filename)
    record = load_file(record_loc)
    if channel is None:
        for channel_ in range(len(record["Data"])):
            plot_signal(record, channel=channel_,filename=os.path.splitext(filename)[0], normalize = True)
    else:
        plot_signal(record, channel=channel,filename=os.path.splitext(filename)[0], normalize = True)

   
#filename = "brux2.edf"
#plot_file_signals(folder,filename, channel = None)
        
###########################################################
#
# These functions are for creating time synced data tables
#
###########################################################
        
import pandas as pd
import datetime as dt

def get_frequency(record_dict,channel):
    """
    A method used to return the parameters necessary to resample a data series in timestamp format
    """
    length = record_dict["FileDuration"]
    samples = len(record_dict["Data"][channel])

    if record_dict["SignalHeaders"][channel]["sample_rate"] != samples/length:
        print("Sample rate mismatch. Expected",str(record_dict["SignalHeaders"][channel]["sample_rate"]),"but have",str(samples/length))

    frequency = str(int(10**9*length/samples))+"N"
    
    return frequency, length, samples

def generate_series(record_dict, channel):
    """
    Returns data for the selected channel with a timestamp index based on the signal sample rate
    """
    gain = (record_dict["SignalHeaders"][channel]["physical_max"]-record_dict["SignalHeaders"][channel]["physical_min"])/(record_dict["SignalHeaders"][channel]["digital_max"]-record_dict["SignalHeaders"][channel]["digital_min"])
    data = record_dict["Data"][channel] * gain

    start_date = record_dict["Header"]["startdate"]
    frequency, length, samples = get_frequency(record_dict,channel)
    index = pd.date_range(start_date,periods=samples,freq=frequency)
    
    if record_dict["SignalHeaders"][channel]["label"] in record_dict["SignalHeaders"][channel]["transducer"]:
        name = record_dict["SignalHeaders"][channel]["transducer"]
    else:
        name = record_dict["SignalHeaders"][channel]["transducer"] + " - " + record_dict["SignalHeaders"][channel]["label"]
    return pd.Series(data=data,index=index, name=name)

def generate_features(record_dict):
    """
    Generate the features table
    """
    channels = len(record_dict["SignalHeaders"])
    table = pd.DataFrame()
    for i in range(channels):
        table = pd.concat((table,generate_series(record_dict, channel=i)),axis=1, join="outer")
    return table

def expand_data_labels(data):
    """This function expands the data labels by allocating new labels at timestamps implied by the duration
    column."""
    new_data = []
    no_position = False
    for position in range(len(data)):
        #['Sleep Stage', 'Position', 'Event', 'Duration[s]', 'Location']
        temp_stage = data["Sleep Stage"][position]
        try:
            temp_position = data["Position"][position]
        except:
            temp_position = None
            no_position = True
        temp_location = data["Location"][position]
        temp_duration = data["Duration[s]"][position]
        temp_time_tot = data.index[position]
        temp_type_ar = data["Event"][position]
        temp_priority = int("MCAP" not in temp_type_ar)
        temp_index = temp_time_tot + dt.timedelta(seconds=temp_priority/2)

        for i in range(data["Duration[s]"][position]):
            if temp_position == None:
                new_data.append([temp_time_tot,temp_stage,temp_type_ar,temp_duration,temp_location, temp_priority, temp_index])
            else:
                new_data.append([temp_time_tot,temp_stage,temp_position,temp_type_ar,temp_duration,temp_location, temp_priority, temp_index])
            temp_duration = temp_duration - 1
            temp_time_tot = temp_time_tot + dt.timedelta(seconds=1)
            temp_index = temp_index + dt.timedelta(seconds=1)
    
    # Converting list to dataframe
    new_data = pd.DataFrame(new_data).set_index(0)
    del new_data.index.name
    
    #Removing duplicated timestamps by keeping the label that was called in first for overlapping seconds only
    if no_position:
        new_data = new_data.sort_values(by=[5,6], ascending=True, kind="quicksort")
    else:
        new_data = new_data.sort_values(by=[6,7], ascending=True, kind="quicksort")
    new_data = new_data[~new_data.index.duplicated(keep="first")]
    
    #Removing ancillary columns and renaming return columns
    if no_position:
        new_data = new_data[list(range(1,5))]
    else:
        new_data = new_data[list(range(1,6))]
    new_data.columns = data.columns
    
    return new_data.sort_index(axis=0)


def generate_labels(labels_loc, table, redate=True, expand_labels = True):
    """
    Generate the labels table with timestamps compatible with those of the feature table
    """
    skip = 0
    error =False
    # A temp function for datetime conversion
    def time_conv(labels,t):
        return dt.datetime.strptime(labels.index[t], "%H:%M:%S")
    
    if os.path.isfile(labels_loc):
        try_again = True
        while try_again:
            try:
                labels = pd.read_csv(labels_loc,sep="\t",skip_blank_lines=True,header="infer",skiprows=skip) #,index_col=2
                
                try:
                    labels.set_index("Time [hh:mm:ss]", inplace=True); #'Time [hh:mm:ss]'
                except:
                    pass
                if len(labels.columns) <= 2:
                    try_again = True
                    skip += 1
                else:
                    try_again = False
                    print(labels.columns)
            except:
                skip += 1
                if skip >=100:
                    print("Labels Parser -> Error finding table in file. Quitting.")
                    try_again = False
                    error = True
        if not error:
            if redate:
                del labels.index.name
    
                new_index = [time_conv(labels,0)]
                offset_days = (table.index[0] - new_index[0]).days + 1
                new_index[0] += dt.timedelta(days=offset_days)
            
                for i in range(1,len(labels.index)):
                    if time_conv(labels,i).hour < time_conv(labels,i-1).hour:
                        offset_days += 1
                    new_index.append(time_conv(labels,i) + dt.timedelta(days=offset_days))

                labels.index = new_index

            if expand_labels:
                return expand_data_labels(labels)
            else:
                return labels
        else:
            return None

def generate_table(filename, folder, save_csv = False):
    """
    This is the final output method that either returns features and labels together, or dumps
    the same o a (very very big) csv file.
    """
    record_loc = os.path.join(folder,filename)
    labels_loc = os.path.join(folder,filename.split(".")[0]+".txt")
    record_dict = load_file(record_loc)
    table = generate_features(record_dict)
    labels = generate_labels(labels_loc, table, redate=True, expand_labels = True)
    final = pd.merge_asof(table,labels,left_index=True, right_index=True, direction="nearest") #
    if save_csv:
        csv_file_loc = os.path.join(folder,"csv",filename.split(".")[0]+".csv")
        final.to_csv(csv_file_loc)
        return None
    else:
        return final
    
def process_files_and_csv(folder):
    """
    Go over all the files and process them to csv files (could be very time consuming)
    """
    errors = []
    for file in os.listdir(folder):
        filename, extension = os.path.splitext(file)
        if extension == ".edf":
            if not os.path.isfile(os.path.join(folder,"csv",filename+".csv")):
                #record_loc = os.path.join(folder,file)
                try:
                    print("\n========================\nProcessing:",file)
                    generate_table(file, folder, save_csv = True)
                except:
                    print("Error processing file:",file)
                    errors.append(file)
            else:
                print("\n========================\nFile already exists:",file)
    return errors

#brux2 = generate_table("brux2.edf", folder, save_csv = False)
    
###########################################################
#
# These functions are for dealing with frequency filtering
#
###########################################################
    
import scipy.signal as signal

def normalizer(data, sigma = 10, value = 0):
    """Use this to normalize the columns of a numpy array by making the values fall in the range of [-1,1]
    and remove outliers that fall outside sigma number of standard deviations, and replace those values with
    value. Use value = -1 to replace with saturation value."""
    print("Normalizing features between range, excluding outliers greater than",str(sigma),"standard deviations.")
    def reject_outlier(data, sigma=3, value = -1):
        for i in range(len(data[0])):
            temp = data[:,i]
            if value == -1:
                temp[abs(temp - np.mean(temp)) > sigma * np.std(temp)] = np.sign(temp) * sigma * np.std(temp)
            else:
                temp[abs(temp - np.mean(temp)) > sigma * np.std(temp)] = value
            data[:,i] = temp
        return data

    def normalize(data):
        for i in range(len(data[0])):
            data[(slice(None),i)] /= np.max(np.abs(data[(slice(None),i)]))
        return data

    data = normalize(data)
    
    data = reject_outlier(data, sigma = sigma, value = value)
    
    data = normalize(data)
    
    return data

def apply_butter_band_filter(data, low, high, fs, order = 2, bt="band"):
    nyq = fs / 2
    b, a = signal.butter(order, [low/nyq, high/nyq], btype=bt, analog=False)
    return signal.filtfilt(b, a, data)

def split_EEG_bands(vector, fs, apply_low_pass = "cheby", cheby_rp = 12, cutoff_hz = 30):
    print("Low pass filter:",apply_low_pass,"Filter ripple (db):",cheby_rp,"Cutoff freq for low pass:",cutoff_hz)
    data = np.array(vector)
    if apply_low_pass == "cheby":
        # Using a cheby1 filter as an antialiasing filter with a max ripple of 5 db (or whatever rp)
        b, a = signal.cheby1(2, cheby_rp, cutoff_hz/128, btype = "lowpass", output="ba")
        data = signal.filtfilt(b, a, data)
    if apply_low_pass == "butter":
        # Using a butter stop filter for AAF. Not sure if this works well.
        data = apply_butter_band_filter(data, 0.5, cutoff_hz, fs, bt="stop")
    if apply_low_pass =="fir":
        nyq_rate = fs/2
        # Here cherby_rp is the attenuation beyond cutoff_hz
        # 2 Hz is the transition width for the cutoff
        N, beta = signal.kaiserord(cheby_rp, 2*nyq_rate)
        taps = signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        data = signal.lfilter(taps, 1.0, data)
        
    delta1 = apply_butter_band_filter(data, 0.5, 4.0, fs)
    #delta1b = apply_butter_band_filter(data, 2.0, 4.0, fs)
    theta1 = apply_butter_band_filter(data, 4.0, 8.0, fs)
    alpha1 = apply_butter_band_filter(data, 8.0, 12.0, fs)
    sigma1 = apply_butter_band_filter(data, 12.0, 15.0, fs)
    beta1 = apply_butter_band_filter(data, 15.0, 30.0, fs)
    #gamma1 = apply_butter_band_filter(data, 30.0, 80.0, fs)
    
    # Square the signals
    delta1 = np.power(np.abs(delta1),2)
    theta1 = np.power(np.abs(theta1),2)
    alpha1 = np.power(np.abs(alpha1),2)
    sigma1 = np.power(np.abs(sigma1),2)
    beta1 = np.power(np.abs(beta1),2)
    
    #Creating table
    bands = np.c_[delta1,theta1,alpha1,sigma1,beta1]
    
    # Normalizing bands between -1 and 1
    bands = normalizer(bands, sigma = 10, value = 0)
    
    # Repackaging
    
    bands = pd.DataFrame(bands)
    bands.columns = ["delta","theta","alpha","sigma","beta"]
    bands.index = vector.index
    
    # Resampling the data using the average value per each (non-overlapping) second
    bands = bands.resample("1S").mean()
    
    # Band descriptors
    #bands["delta"]
    
    # Adding multi-index identifier
    bands.columns = pd.MultiIndex.from_product([[vector.name], bands.columns], names=["probes", "bands"])
    
    return bands

def generate_nonEEG_bands(vector):
    """This function squares and normalizes a signal without separating into frequency bands."""
    result = np.power(np.abs(np.array(vector)),2)
    result = normalizer(result, sigma = 10, value = 0)
    
    result = pd.DataFrame(result)
    result.columns = ["value"]
    result.index = vector.index
    
    #result = result.resample("1S").mean()
    
    return result

def generate_features_bands(record_dict):
    """
    Similar to generate_features() but outputs the bands at a resampled rate of 1 Hz instead.
    """
    table = generate_features(record_dict)
    new_table = pd.DataFrame()
    for i, col in enumerate(table.columns):
        if "EEG" in col:
            new_feature = split_EEG_bands(table[col], fs = record_dict["SignalHeaders"][i]["sample_rate"], 
                                          apply_low_pass = "fir", cheby_rp = 12, cutoff_hz = 32)
        else:
            # If the column is not EEG data, still we need to resample and make headers compatible
            new_feature = table[col]
            new_feature = new_feature.resample("1S").mean()
            new_feature.columns = pd.MultiIndex.from_product([[new_feature.name], ["Value"]], names=["probes", "bands"])
        try:
            new_table = pd.concat([new_table,new_feature], axis=1)
        except:
            new_table = new_table.append(new_feature)
            print("Error adding data column",str(i),"labeled",col)
    #new_table = pd.concat(new_table, axis=1)
    
    
    return new_table

def generate_table_bands(filename, folder, save_csv = False, load_csv=True, load_edf = False):
    """
    This is the final output method that either returns band features and labels together, and at the same time
    is able to store a copy of the file to the hard drive for easy retrieval.
    """
    csv_filename = os.path.join(folder,"csv",filename.split(".")[0]+"_bands.csv")
    if load_csv and os.path.isfile(csv_filename):
        final = pd.read_csv(csv_filename,index_col=0,sep=",",header=[0,1], parse_dates=False, infer_datetime_format=False,keep_date_col=False)
    else:
        record_loc = os.path.join(folder,filename)
        labels_loc = os.path.join(folder,filename.split(".")[0]+".txt")
        record_dict = load_file(record_loc, try_convertion = load_edf)
        table = generate_features_bands(record_dict)
        labels = generate_labels(labels_loc, table, redate=True, expand_labels = True)
        final = pd.merge_asof(table,labels,left_index=True, right_index=True, direction="nearest") #
        # Sometimes the headers get messed up
        try:
            final.columns.levels
        except:
            final.columns = pd.MultiIndex.from_tuples(tuple([col if type(col) == tuple else (col,"value") for col in final.columns]))
        if save_csv:
            final.to_csv(csv_filename)
    return final

def generate_all_table_bands(folder):
    errors = []
    for file in os.listdir(folder):
        filename, extension = os.path.splitext(file)
        if extension == ".edf":
            if not os.path.isfile(os.path.join(folder,"csv",filename+"_bands.csv")):
                try:
                    file_loc = os.path.join(folder,file)
                    print("\n========================\nProcessing:",file)
                    t = generate_table_bands(file_loc, folder, save_csv = True, load_csv=False, load_edf = True)
                    print(t.info())
                except:
                    print("Error processing file:",file)
                    errors.append(file)
            else:
                print("\n========================\nFile already exists:",file)
    return errors


