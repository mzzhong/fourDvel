#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

from fourdvel import fourdvel

import numpy as np
import os

class display(fourdvel):

    def __init__(self):
        super(display,self).__init__()

    def show_est(self,show_dict):

        # Write to txt file.
        est_xyz = self.est_dict_name + '.xyz'
        f = open(os.path.join(self.design_mat_folder,est_xyz),'w')
        cap=100
        for key in sorted(show_dict.keys()):
            lon, lat = key
            value = show_dict[key]
            value = min(value,cap)
            if value == 0:
                value == 100

            if not np.isnan(value):
                f.write(str(lon)+' '+str(lat)+' '+str(min(value,cap))+'\n')

        f.close()

        return
     
        # Prepare the matrix.

        # First coordinate (width)
        lon0 = -84
        lon_interval = 0.02
        lon1 = -68
        lon_list = np.arange(lon0, lon1, lon_interval)

        # Second coordinate (length) 
        lat0 = -74.5
        lat_interval = -0.005
        lat1 = -77.5
        lat_list = np.arange(lat0, lat1, lat_interval)

        llon, llat = np.meshgrid(lon_list,lat_list)

        showmat = np.zeros(llon.shape)
        showmat[:] = np.nan

        #print(showmat.shape)

        points={}

        for key in show_dict.keys():
            lon,lat = key
            ind_x = int(round((lon-lon0)/lon_interval))
            ind_y = int(round((lat-lat0)/lat_interval))

            if not (ind_y,ind_x) in points.keys():
                points[(ind_y,ind_x)] = (lon,lat)
            else:
                print('Confliction!')
                print(points[(ind_y,ind_x)])
                print(key)
                print(ind_x,ind_y)
                print(stop)

            if ind_x>=0 and ind_x<len(lon_list) and ind_y>=0 and ind_y<len(lat_list):
                showmat[ind_y,ind_x] = show_dict[key]

        print(len(points.keys()))

        # Plot.
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)

        extent=[lon_list[0],lon_list[-1],lat_list[-1],lat_list[0]]

        f1 = ax.imshow(showmat,cmap=plt.cm.coolwarm,vmin=0,vmax=0.15, extent=extent, aspect='auto')
        #f1 = ax.imshow(showmat,cmap=plt.cm.jet,vmin=0,vmax=10)

        # Plot grounding line.
        f = open(self.glfile)
        data = f.readlines()
        print(len(data))

        #subdata = [pair for pair in data if float(pair.split()[0])>=lon0 and float(pair.split()[0])<=lon1 and float(pair.split()[1])<=lat0 and float(pair.split()[1])>=lat1 ]
        #print(len(subdata))
        
        #x = [pair.split()[0] for pair in subdata]
        #y = [pair.split()[1] for pair in subdata]

        x=[]
        y=[]
        for i in range(0,len(data),100):
            pair = data[i]
            lon = float(pair.split()[0])
            lat = float(pair.split()[1])
            x.append(lon)
            y.append(lat)
            
        ax.plot(x,y,'k.',label='grouding line')
        ax.legend(loc=3)

        ax.set_xlim([lon_list[0],lon_list[-1]])
        ax.set_ylim([lat_list[-1],lat_list[0]])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')


        #ax.set_title('sensitivity of noise')
        cbar=fig.colorbar(f1,orientation='vertical',shrink=0.8,ticks=np.arange(0,0.201,0.05))
        cbar.set_label('max value in sensitivity matrix', fontsize=15)

        #fig.colorbar(f1,orientation='horizontal',shrink=0.8)
        fig.savefig('temp.png',format='png')
        
        print('Done')

        return 0

    def write_dict_to_xyz(self, show_dict, xyz_name):
        
        # Write to txt file.

        f = open(xyz_name,'w')
        cap=100
        for key in sorted(show_dict.keys()):
            lon, lat = key
            value = show_dict[key]

            # More then one value (velocity vector).
            if isinstance(value,tuple):
                if not np.isnan(value[0]):
                    record = str(lon)+' '+str(lat)
                    for irec in range(len(value)):
                        record = record + ' ' + str(value[irec])
                    record = record + '\n'
                    f.write(record)
            else:

                # Only output valid values.
                value = min(value,cap)
                if not np.isnan(value):
                    f.write(str(lon)+' '+str(lat)+' '+str(value)+'\n')

        f.close()
        return
 
    def tide_symbols(self,tide_name):
        pass


    def display_fitting(self, point, data_info, offsetfields, design_mat, data_vec, model_vec, tide_vec):

        G = design_mat
        m = model_vec
        pred = np.matmul(G,m)
        obs = data_vec

        print(offsetfields)
        print(tide_vec)
        print(self.modeling_tides)
        print(self.n_modeling_tides)

        print('continous prediction')
        t_axis = np.arange(-600,600,0.0005)

        con_pred = np.zeros(shape=t_axis.shape)

        for k, tide_name in enumerate(self.modeling_tides):
            for t in range(3):
                if t==2:
                    ampU = tide_vec[3+k*6+t]
                    phaseU = tide_vec[3+k*6+t+3]
                    omega = 2*np.pi / self.tide_periods[tide_name]

                    dis_ampU = self.velo_amp_to_dis_amp(ampU, tide_name)
                    dis_phaseU = self.velo_phase_to_dis_phase(phaseU, tide_name)

                    print(tide_name, dis_ampU)

                    con_pred = con_pred + dis_ampU * np.sin(omega*t_axis + dis_phaseU)

        print(stop)

        #################
        N = len(pred)
        print(data_info, N)
        print(offsetfields, len(offsetfields))

        # Get number of tracks.
        Nt = 0
        valid_tracks = []
        for i, info in enumerate(data_info):
            # Not empty:
            if info[1]>0:
                Nt += 1
                valid_tracks.append(info)


        print('valid_tracks: ',valid_tracks)
        j=0
        k=0
        for i, info in enumerate(valid_tracks):
            k += info[1]

            # Only range offset
            N = k-j

            #ax = fig.add_subplot(Nt,1,i+1)

            fig = plt.figure(1,figsize=(15,10))
            # Continous data.
            ax = fig.add_subplot(211)
            ax.plot(t_axis, con_pred,color='k')
            ax.set_xlim([0,self.tide_periods['Msf']])
            ax.set_xlim([0,60])

            ax = fig.add_subplot(212)
            for i, offsetfield in enumerate(offsetfields[j:k]):

                t_a = (offsetfield[0] - self.t_origin.date()).days + offsetfields[i][4]
                t_b = (offsetfield[1] - self.t_origin.date()).days + offsetfields[i][4]

                pred_offset = pred[(j+i)*2]
                obs_offset = obs[(j+i)*2]

                misfit = abs(pred_offset - obs_offset)

                t_a = t_a % self.tide_periods['Msf']
                t_b = t_b % self.tide_periods['Msf']

                if t_a < t_b:
                    ax.plot([t_a,t_b],[misfit,misfit],'k')
                else:
                    ax.plot([t_a,self.tide_periods['Msf']],[misfit,misfit],'k')
                    ax.plot([0,t_b],[misfit,misfit],'k')
            
            ax.set_xlim([0,self.tide_periods['Msf']])

            # Show the dates.
            # Show the data
            #ax.plot(range(N), pred[2*j:2*k:2], color='r', marker='.', markersize=5, linestyle='', label='pred')
            #ax.plot(range(N),obs[2*j:2*k:2],color='k',marker='.', markersize=5, linestyle='', label='obs')
            #ax.legend(loc=1)

            title = str(info[0][0])+'_'+str(info[0][1])
            ax.set_title(title)

            j+= info[1]

            fig.savefig(title + '.png')
            plt.close()

        print(stop)
                

    def display_vecs(self, stacked_vecs, row_names, column_names, label):

        # The unit of tide_vec is m/d.
        # The method takes care of unit conversion:
        # Velocity to amplitudes.
        # meter to centimeter.
        
        # For the three components.
        modeling_tides = self.modeling_tides

        n_values, n_vecs = stacked_vecs.shape

        for comp in range(3):

            comp_name = self.comp_name(comp)
            print(comp_name, ':')

            ## Generate column names in table format.
            columns_all = []
            for column_name in column_names:
                if column_name == 'Secular':
                    columns_all.append('$Secular\ velocity\ (cm/d)$')
                else:
                    tide_name = column_name
                    column_symbol = '$' + tide_name[0] + '_{' + tide_name[1:] + '}\ (cm)$'
                    columns_all.append(column_symbol)
    
                n_cols = 1 + self.n_modeling_tides
                columns = columns_all[0:n_cols+1]

            # Number of rows for all tides.
            rows = row_names
            n_rows = len(rows)

            ##########  Amplitude ###################
            cell_values = np.zeros(shape=(n_rows,n_cols))
            cell_text = []
            # For synthetic and estimated values.
            for row in range(n_rows):

                vec = stacked_vecs[:,row]

                # Find the amplitude of all constituents.
                col_values = np.zeros(shape=(n_cols,))
                col_text = []
                for col in range(n_cols):
                    # Secular velocity.
                    if col==0:
                        col_values[col] = vec[comp]

                    # Tides, convert to displacement.
                    else:
                        col_values[col] = vec[int(col>=1)*(3 + 6*(col-1)) + comp]
                        col_values[col] = self.velo_amp_to_dis_amp(col_values[col],modeling_tides[col-1])
                    
                    col_values[col] = self.float_rounding(self.m2cm(col_values[col]),100)
                    col_text.append('%.2f' % col_values[col])

                cell_text.append(col_text)
                cell_values[row,:] = col_values

            #df = pd.DataFrame(cell_values, index=rows, columns=columns)

            # Show the table.
            fig = plt.figure(1,figsize=(20,5))
            plt.clf()
            fig.suptitle(comp_name)

            ax = fig.add_subplot(211)
            ax.axis('off')
            ax.axis('tight')
            ax.set_title('Amplitude')

            #print(cell_text)
            #print(columns)
            #print(rows)

            the_table = ax.table(cellText=cell_text, colLabels= columns, rowLabels = rows,loc='center')
            #the_table = ax.table(cellText=cell_text, colLabels= columns, loc='center')

            #the_table.set_fontsize(30)
            the_table.scale(1,2)

            # Change text alignment.
            cells = the_table.get_celld()
            print(cells)
            
            #for ir in range(n_rows+1):
            #    for ic in range(-1,n_cols):
            #        if not (ir==0 and ic==-1):
            #            if ic>-1:
            #                cells[ir,ic].set_width(0.1)
            #                cells[ir,ic]._loc = 'center'

            #            else:
            #                pass
            #                #cells[ir,ic].set_width(0.01)
            #                #cells[ir,ic]._loc = 'right'

            #print(cell_text)

            ################  Phase ################################
            columns_all = []
            for column_name in column_names:
                if column_name == 'Secular':
                    columns_all.append('$Secular\ velocity\ (degree)$')
                else:
                    tide_name = column_name
                    column_symbol = '$' + tide_name[0] + '_{' + tide_name[1:] + '}\ (degree)$'
                    columns_all.append(column_symbol)

            columns = columns_all[0:n_cols+1]

            # Threshold for saying phase value is meaningful.
            threshold = 0
            cell_values = np.zeros(shape=(n_rows,n_cols))
            cell_text = []
            # For synthetic and estimated values.
            for row in range(n_rows):

                vec = stacked_vecs[:,row]

                # Find the phases of all constituents.
                col_values = np.zeros(shape=(n_cols,))
                col_text = []
                for col in range(n_cols):
                    
                    # Phase of secular velocity.
                    if col == 0:
                        col_values[col] = np.nan
                    else:

                        # Tides.
                        velo_amp = vec[3 + 6*(col-1) + comp]
                        velo_amp = self.float_rounding(self.m2cm(velo_amp),100)

                        # If velocity_amp is large enough.
                        #if velo_amp>threshold:
                        phase_value = vec[3 + 6*(col-1) + comp + 3]
                        
                        if rows[row] != 'Uncertainty':
                            phase_value = self.velo_phase_to_dis_phase(phase_value)
                            phase_value = self.wrapped(phase_value)
                        
                        phase_value = self.rad2deg(phase_value)
                        phase_value = self.float_rounding(phase_value,100)
                        col_values[col] = phase_value

                    col_text.append('%.2f' % col_values[col])

                cell_text.append(col_text)
                cell_values[row,:] = col_values

            #df = pd.DataFrame(cell_values, index=rows, columns=columns)

            # Show the table.
            fig = plt.figure(1)

            ax = fig.add_subplot(212)
            ax.axis('off')
            ax.axis('tight')
            ax.set_title('Phase')

            the_table = ax.table(cellText=cell_text, colLabels= columns, rowLabels = rows,loc='center')
            
            #the_table.set_fontsize(30)
            the_table.scale(1,2)

            # Change text alignment.
            #cells = the_table.get_celld()
            #for ic in range(n_cols):
            #    for ir in range(n_rows+1):
            #        cells[ir,ic]._loc = 'center'

            #plt.show()
            plt.savefig(os.path.join('/home/mzzhong/insarRoutines/fig_sim',label+'_'+comp_name[0]+'.png'), format='png')

            plt.close()

        return 0

    def show_resid_dist(self, resid, label):

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(121)
        ax.hist(resid[0:-1:2])
        ax.set_title('range') 

        ax = fig.add_subplot(122)
        ax.hist(resid[1:-1:2])
        ax.set_title('azimuth')

        plt.savefig(os.path.join('/home/mzzhong/insarRoutines/fig_sim',label+'_'+'resid.png'), format='png')

        plt.close()

    def show_model_mat(self,showmat):

        n_rows, n_cols = showmat.shape

        vmax = 1
        vmin = -1

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        #im = ax.imshow(showmat, cmap = cm.jet, vmax=vmax, vmin=vmin)

        im = ax.imshow(showmat, cmap = cm.jet)
 
        fig.colorbar(im,orientation='horizontal',shrink=0.7)

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_cols))

        modeling_tides = self.modeling_tides
        tide_periods = self.tide_periods

        it = 0
        for tide_name in modeling_tides:

            ax.text(n_cols+1, 3+(it+1)*6-3, tide_name + ': ' + str(tide_periods[tide_name]), fontsize=10)
            
            ax.text(n_cols+1, 3+(it+1)*6-2,str(3+it*6) + ' - ' + str(3+(it+1)*6-1),fontsize=10)

            ax.plot(np.asarray(range(n_cols)), (3+it*6-0.5) + np.zeros(shape=(n_cols,)),linestyle='--',color='k')
        
            it = it + 1

        fig.savefig('./fig_sim/model_mat.png',format='png')
        
        return 0

