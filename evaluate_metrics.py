import scipy.io
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

def get_scaling_factor(GT,mode='None'):
    if mode=='None':
        scaling_factor=1.0
    elif mode=='Polynomial':
        scaling_factor=-0.148*GT*GT+1.29*GT-0.16649
    elif mode=='Log':
        scaling_factor=1.059*np.log(GT)+1.142
    else :
        print('Wrong input for scaling mode, choose between : Polynomial and None')
    return scaling_factor

def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(description='Evaluate metrics')
    parser.add_argument("--prediction_file", "-p", type=str, default='wheelchair',
                        help='files where the prediction is stored')
    parser.add_argument("--file_number", "-f", type=int, default=2,
                        help='files number in hard drive')
    parser.add_argument("--scaling_mode", "-s", type=str, default='None',
                        help='Scaling estimation used, choose between : Polynomial and None')
    parser.add_argument("--output_file", "-o", type=str, default='metrics_data',
                        help='ouptut files for the metrics')
    parser.add_argument("--create_file", "-cf", type=bool, default=False,
                        help='Create metrics files')
    parser.add_argument("--relative_position", "-rp", type=bool, default=False,
                        help='Return the relative error of the joint in respect to the neck')
    parser.add_argument("--rescaled", "-rs", action='store_true',
                        help='Return the relative error of the joint in respect to the neck')
    args = parser.parse_args()
    return args
data_file='label_dist'
def main():
    args = cli()
    ascii_esc=27
    #Gets files
    hard_drive_path='/media/drame/INTENSO2/pds/dataset/custom/'+data_file+'/'
    #hard_drive_path='/media/drame/INTENSO/pds/dataset/custom/wheelchair/'
    file_number=args.file_number
    scaling_mode=args.scaling_mode
    output_files='metrics/'+args.output_file
    print('\n ----------------------------------------------------------------------- \n Read files :{}'.format(file_number))
    GT_file=hard_drive_path+'data{}.json'.format(file_number)
    estimation_file='output/{}/'.format(args.prediction_file)+data_file+'{}.json'.format(file_number)

    f = open(GT_file,)
    depth_GT = json.load(f)

    RMS=[]
    abs_rel=[]
    prediction_mean=[]
    GT_mean=[]
    estimation = []

    for idx,line in enumerate(open(estimation_file, 'r')):
        #print(idx,'-----------------------------')
        #break
        estimation=(json.loads(line)) 
        if(len(estimation["predictions"])!=0):
            #print(len(estimation["predictions"]))
            keypoints=np.reshape((estimation["predictions"][0]["keypoints"]),(-1,3)).astype(int)
            depth_pred=np.array((estimation["depth_pred"]))
            GT=np.array(depth_GT["frame{}".format(idx+151)])/1000

            #Midas produced a rescaled disparity maps, we rescale GT to reproduce metrics from the paper
            if args.rescaled==True:
                t=np.median(GT)
                s=1/np.size(GT)*(np.sum(np.abs(GT-t)))
                GT=(GT-t)/s


            for depth_pred_person in depth_pred:
                RMS_temp=[]
                prediction_mean_tmp=[]
                GT_mean_tmp=[]
                abs_rel_tmp=[]
                for i,keypoint in enumerate(keypoints):
                    keypoint[1]=keypoint[1]-1
                    keypoint[0]=keypoint[0]-1
                    scaling_factor=get_scaling_factor(GT[keypoint[1]][keypoint[0]],scaling_mode)
                    KP_pred=depth_pred_person[i]
                    KP_GT=GT[keypoint[1]][keypoint[0]]
                    #print('keypoints',keypoint[1],keypoint[0],'GT',GT[keypoint[1]][keypoint[0]], 'pred',depth_pred_person[i])
                    if (KP_GT!=0.0) and np.isnan(KP_pred)==False:
                        prediction_mean_tmp.append(KP_pred)
                        KP_pred=KP_pred*scaling_factor

                        GT_mean_tmp.append(KP_GT)
                        RMS_temp.append(np.sqrt(np.square(KP_GT-KP_pred)))
                        abs_rel_tmp.append((KP_GT-KP_pred)/(KP_GT))
            prediction_mean_tmp=np.nanmean(prediction_mean_tmp)
            GT_mean_tmp=np.nanmean(GT_mean_tmp)
            RMS_temp=np.nanmean(RMS_temp)
            abs_rel_tmp=np.nanmean(abs_rel_tmp)
            scaling_factor_approx=GT_mean_tmp/(prediction_mean_tmp)

            prediction_mean.append(prediction_mean_tmp)
            GT_mean.append(GT_mean_tmp)
            RMS.append(RMS_temp)
            abs_rel.append(abs_rel_tmp)

            if GT_mean_tmp<=6.0 and GT_mean_tmp>=0.0:
                df_tmp=pd.DataFrame([[idx+1,RMS_temp,abs_rel_tmp,GT_mean_tmp,prediction_mean_tmp,scaling_factor_approx,file_number]],
                                index = ['frame{}'.format(idx+1)], 
                                columns = ['frame','RMS', 'abs rel','mean GT','mean pred','scaling factor','file number'])
            else :
                df_tmp=pd.DataFrame([[idx+1,'NaN','NaN','NaN','NaN','NaN',file_number]],
                            index = ['frame{}'.format(idx+1)], 
                            columns = ['frame','RMS', 'abs rel', 'mean GT','mean pred','scaling factor','file number'])

        else:
            df_tmp=pd.DataFrame([[idx+1,'NaN','NaN','NaN','NaN','NaN',file_number]],
                            index = ['frame{}'.format(idx+1)], 
                            columns = ['frame','RMS', 'abs rel', 'mean GT','mean pred','scaling factor','file number'])

        if idx==0:
            df=df_tmp
        else:
            df=df.append(df_tmp)
    print('writting')
    if args.create_file==True:
        df.to_csv('{}.csv'.format(output_files),mode='w')
    else:
        df.to_csv('{}.csv'.format(output_files),mode='a', header=False)

    f.close()


if __name__ == '__main__':
    main()
#print('rms',RMS, '\n mean RMS in meters',np.mean(RMS))


