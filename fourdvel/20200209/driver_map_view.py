#!/usr/bin/env python3
import os
import sys

import argparse

def createParser():
    parser = argparse.ArgumentParser( description='control the gmt plotting')

    parser.add_argument('-s','--subject', dest='subject',type=str,help='subject for plotting (secular, tide, resid, speedmap, basemap, dem, temporary)',default=None, required=True)

    parser.add_argument('-n','--test_id', dest='test_id',type=str,help='id of the test for ploting',default=None)

    parser.add_argument('-p','--prefix', dest='prefix',type=str,help='true or estimation',default='est')

    parser.add_argument('-c','--comp', dest='comp',type=str,help='component of data',default=None)

    parser.add_argument('-k','--kind', dest='kind',type=str,help='kind of data',default=None)

    parser.add_argument('-tc','--tocm', dest='tocm',type=str,help='convert to cm',default='0')

    parser.add_argument("--name", dest='name',type=str,help="provide additional name",default="temp")

    return parser

def cmdLineParse(iargs = None):

    parser = createParser()

    return parser.parse_args(args=iargs)


class map_view():

    def __init__(self):

        result_folder="/home/mzzhong/insarRoutines/estimations"

        self.test_id = 236
        self.dem = 1
        self.evans = 1

        self.this_result_folder = os.path.join(result_folder,str(self.test_id))

        self.value_range()

        self.tide_list =['M2','O1','Msf','Mf']

    def write_cmd(self, dem=0, evans=0, speedmap=0, test_id='nan', name='nan', 
                        comment="Evans", minval='nan', maxval='nan',unit='nan',title="Evans",velo='nan', other_1=0, temporary=0, tocm='0'):


        if tocm == '1':
            if minval!='nan':
                minval = str(float(minval)*100)
            if maxval!='nan':
                maxval = str(float(maxval)*100)
            if unit!='nan':
                unit = 'c'+unit

        cmd=' '.join(['./map_view_2.gmt',
                                        '-b', str(dem), 
                                        '-e', str(evans),
                                        '--speedmap', str(speedmap),
                                        '--id',str(test_id),
                                        '--name', name, 
                                        '-c', comment, 
                                        '--minval', str(minval),
                                        '--maxval', str(maxval), 
                                        '-u', unit, 
                                        '-t', title,
                                        '--velo', velo,
                                        '--other_1', str(other_1),
                                        '--temporary', str(temporary),
                                        '--tocm', str(tocm)])
        return cmd

    def gen_comment(self):

        if self.prefix == 'true':
            comment='true'
            print(comment)
        else:
            comment="estimation_{}".format(self.test_id)
            print(comment)

        return comment
    
    def get_velofile(self):
 
        velofile = os.path.join("/home/mzzhong/insarRoutines/estimations",
                                str(self.test_id),'_'.join([str(self.test_id),self.prefix,'secular','horizontal','velocity']))

        velofile = velofile + '.xyz'

        print(velofile)

        return velofile


    def show_secular(self):

        name = '_'.join([str(self.test_id), self.prefix, self.subject, self.comp, self.kind])

        comment = self.gen_comment() 

        if 'velocity_difference' in self.kind:
            minval, maxval = self.dict_range[(self.prefix, self.subject, self.comp, 'velocity_difference')]
        else:
            minval, maxval = self.dict_range[(self.prefix, self.subject, self.comp, self.kind)]

        unit = 'm/day'

        title = name

        if self.comp == 'horizontal' and self.kind == 'velocity':
            velofile = self.get_velofile()
            cmd = self.write_cmd(dem=self.dem, evans=self.evans, test_id=self.test_id,
                            name=name, comment=comment, minval=minval, maxval=maxval,
                            unit=unit, title=title, velo=velofile, tocm=self.tocm)

        else:
            cmd = self.write_cmd(dem=self.dem, evans=self.evans, test_id=self.test_id,
                            name=name, comment=comment, minval=minval, maxval=maxval,
                            unit=unit, title=title, tocm=self.tocm)

        print(cmd)
        os.system(cmd)

        return


    def show_tide(self):
        
        name = '_'.join([str(self.test_id), self.prefix, self.subject, self.comp, self.kind])

        comment = self.gen_comment()

        minval, maxval = self.dict_range[(self.prefix, self.subject, self.comp, self.kind)]

        # Deternine unit.
        if 'amplitude' in name:
            unit = 'm'
        elif 'phase' in name:
            
            if "Msf" in name:
                unit = "day"
            else:
                unit = 'minute'

        title = name

        cmd = self.write_cmd(dem=self.dem, evans=self.evans, test_id=self.test_id,
                            name=name, comment=comment, minval=minval, maxval=maxval,
                            unit=unit, title=title, tocm=self.tocm)

        print(cmd)
        os.system(cmd)

        return


    def show_resid(self):
        
        name = '_'.join([str(self.test_id), self.prefix, self.subject, self.comp])
        comment='nan'

        minval, maxval = self.dict_range[(self.prefix, self.subject, self.comp)]

        unit = 'm'

        title = name
        cmd = self.write_cmd(dem=self.dem, evans=self.evans, test_id=self.test_id,
                            name=name, comment=comment, minval=minval, maxval=maxval,
                            unit=unit, title=title)
        print(cmd)
        os.system(cmd)

        return

    def show_speedmap(self):

        name = 'speedmap'
        title = name
        cmd = self.write_cmd(speedmap=1, name=name)
        print(cmd)
        os.system(cmd)

    def show_other_1(self):

        name = 'other_1'
        title = name
        cmd = self.write_cmd(other_1=1, name=name)
        print(cmd)
        os.system(cmd)

    def show_temporary(self):

        name = self.name
        title = name
        cmd = self.write_cmd(temporary=1, name=name)
        print(cmd)
        os.system(cmd)

    def show_dem(self):
        
        name = 'dem'

        title = name

        cmd = self.write_cmd(dem=1, name=name)
        print(cmd)
        os.system(cmd)

    def show_basemap(self):
        
        name = 'basemap'

        title = name

        cmd = self.write_cmd(name=name)
        print(cmd)
        os.system(cmd)

    def rutford_dict_range(self):

        dict_range = {}

        listA = ['true','est']
        listB = ['uq']

        for item in listA:
            #dict_range [(item,'secular','horizontal','speed')] = (0,2)
            #dict_range [(item,'secular','horizontal','velocity')] = (0,2)

            dict_range [(item,'secular','horizontal','speed')] = (0,1.2)
            dict_range [(item,'secular','horizontal','velocity')] = (0,1.2)
            
            dict_range [(item,'secular','east','velocity')] = [-1,1]
            dict_range [(item,'secular','north','velocity')] = [-2,2]
            dict_range [(item,'secular','up','velocity')] = [-0.1,0.1]

            #### Amplitude ########
            # horizontal.
            dict_range [(item,'Msf','horizontal','displacement_amplitude')] = [0, 0.5]
            dict_range [(item,'Msf','up','displacement_amplitude')] = [0, 1.5]
            dict_range [(item,'Msf','north','displacement_amplitude')] = [0, 0.4]
            dict_range [(item,'Msf','along_flow','displacement_amplitude')] = [0, 0.5]
            dict_range [(item,'Msf','cross_flow','displacement_amplitude')] = [0, 0.5]

            dict_range [(item,'Mf','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'Mf','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'Mf','north','displacement_amplitude')] = [0, 0.2]

            dict_range [(item,'M2','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'M2','north','displacement_amplitude')] = [0, 0.2]

            dict_range [(item,'O1','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'O1','north','displacement_amplitude')] = [0, 0.2]

            # up
            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 2]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.5]

            #### Phase ######
            dict_range [(item,'Msf','along_flow','displacement_phase')] = [-3, 3]
            dict_range [(item,'Msf','cross_flow','displacement_phase')] = [-6, 6]

            dict_range [(item,'Msf','north','displacement_phase')] = [-100, 100]
            dict_range [(item,'Mf','north','displacement_phase')] = [-100, 100]

            dict_range [(item,'M2','up','displacement_phase')] = [-40, 40]
            dict_range [(item,'O1','up','displacement_phase')] = [-120, 120]


            dict_range [(item,'M2','up','displacement_phase_in_deg')] = [-180, 180]
            dict_range [(item,'O1','up','displacement_phase_in_deg')] = [-180, 180]


            #### Residual ######
            dict_range [(item,'resid_of_secular','range')] = [0,1.5]
            dict_range [(item,'resid_of_secular','azimuth')] = [0,1]
            dict_range [(item,'resid_of_tides','range')] = [0,1]
            dict_range [(item,'resid_of_tides','azimuth')] = [0,1]

        for item in listB:
            dict_range [(item,'secular','horizontal','speed')] = (0,0.05)
            dict_range [(item,'secular','east','velocity')] = [0,0.05]
            dict_range [(item,'secular','north','velocity')] = [0,0.05]
            dict_range [(item,'secular','up','velocity')] = [0,0.05]
            dict_range [(item,'Msf','horizontal','displacement_amplitude')] = [0, 0.4]

            dict_range [(item,'Msf','north','displacement_amplitude')] = [0, 0.4]
            dict_range [(item,'Mf','north','displacement_amplitude')] = [0, 0.4]

            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 0.02]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.02]

            dict_range [(item,'M2','horizontal', 'displacement_amplitude')] = [0, 0.2]
            dict_range [(item,'O1','horizontal', 'displacement_amplitude')] = [0, 0.2]


            ## Phase ######
            # In minute.
            dict_range [(item,'Msf','north','displacement_phase')] = [0, 100]
            dict_range [(item,'Mf','north','displacement_phase')] = [0, 100]

            dict_range [(item,'M2','up','displacement_phase')] = [0, 5]
            dict_range [(item,'O1','up','displacement_phase')] = [0, 5]


            dict_range [(item,'M2','up','displacement_phase_in_deg')] = [0, 360]
            dict_range [(item,'O1','up','displacement_phase_in_deg')] = [0, 360]


        dict_range [('est','secular','horizontal','velocity_difference')] = [0, 0.2]

        #print(dict_range)

        return dict_range

    def evans_dict_range(self):

        dict_range = {}

        listA = ['true','est']
        listB = ['uq']

        # Evans
        for item in listA:

            dict_range [(item,'secular','horizontal','speed')] = [0,2]
            dict_range [(item,'secular','horizontal','velocity')] = [0,2]

            dict_range [(item,'secular','east','velocity')] = [-1,1]
            dict_range [(item,'secular','north','velocity')] = [-2,2]
            dict_range [(item,'secular','up','velocity')] = [-0.1,0.1]

            #### Amplitude ########
            # horizontal.
            dict_range [(item,'Msf','horizontal','displacement_amplitude')] = [0, 0.5]
            dict_range [(item,'Msf','up','displacement_amplitude')] = [0, 1.5]
            dict_range [(item,'Msf','north','displacement_amplitude')] = [0, 0.4]
            dict_range [(item,'Msf','along_flow','displacement_amplitude')] = [0, 0.5]
            dict_range [(item,'Msf','cross_flow','displacement_amplitude')] = [0, 0.5]

            dict_range [(item,'Mf','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'Mf','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'Mf','north','displacement_amplitude')] = [0, 0.2]

            dict_range [(item,'M2','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'M2','north','displacement_amplitude')] = [0, 0.2]

            dict_range [(item,'O1','horizontal','displacement_amplitude')] = [0, 0.3]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.8]
            dict_range [(item,'O1','north','displacement_amplitude')] = [0, 0.2]

            # up
            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 2]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.5]

            #### Phase ######
            dict_range [(item,'Msf','along_flow','displacement_phase')] = [-3, 3]
            dict_range [(item,'Msf','cross_flow','displacement_phase')] = [-3, 3]

            dict_range [(item,'Msf','north','displacement_phase')] = [-100, 100]
            dict_range [(item,'Mf','north','displacement_phase')] = [-100, 100]

            dict_range [(item,'M2','up','displacement_phase')] = [-40, 40]
            dict_range [(item,'O1','up','displacement_phase')] = [-120, 120]

            dict_range [(item,'M2','up','displacement_phase_in_deg')] = [60, 120]
            dict_range [(item,'O1','up','displacement_phase_in_deg')] = [60, 120]

            #### Residual ######
            dict_range [(item,'resid_of_secular','range')] = [0,1.5]
            dict_range [(item,'resid_of_secular','azimuth')] = [0,1.5]
            dict_range [(item,'resid_of_tides','range')] = [0,1.5]
            dict_range [(item,'resid_of_tides','azimuth')] = [0,1.5]

        for item in listB:
            dict_range [(item,'secular','horizontal','speed')] = (0,0.05)
            dict_range [(item,'secular','east','velocity')] = [0,0.05]
            dict_range [(item,'secular','north','velocity')] = [0,0.05]
            dict_range [(item,'secular','up','velocity')] = [0,0.05]

            dict_range [(item,'Msf','horizontal','displacement_amplitude')] = [0, 0.4]

            dict_range [(item,'Msf','north','displacement_amplitude')] = [0, 0.4]
            dict_range [(item,'Mf','north','displacement_amplitude')] = [0, 0.4]

            dict_range [(item,'M2','up','displacement_amplitude')] = [0, 0.02]
            dict_range [(item,'O1','up','displacement_amplitude')] = [0, 0.02]

            dict_range [(item,'M2','horizontal', 'displacement_amplitude')] = [0, 0.2]
            dict_range [(item,'O1','horizontal', 'displacement_amplitude')] = [0, 0.2]


            ## Phase ######
            # In minute.
            dict_range [(item,'Msf','north','displacement_phase')] = [0, 100]
            dict_range [(item,'Mf','north','displacement_phase')] = [0, 100]

            dict_range [(item,'M2','up','displacement_phase')] = [0, 5]
            dict_range [(item,'O1','up','displacement_phase')] = [0, 5]

            dict_range [(item,'M2','up','displacement_phase_in_deg')] = [0, 360]
            dict_range [(item,'O1','up','displacement_phase_in_deg')] = [0, 360]


        dict_range [('est','secular','horizontal','velocity_difference')] = [0, 0.2]

        return dict_range


    def value_range(self):

        self.dict_range = self.rutford_dict_range()

        self.dict_range = self.evans_dict_range()

def main(iargs=None):

    mview = map_view()

    inps = cmdLineParse(iargs)

    mview.test_id = inps.test_id
    mview.prefix = inps.prefix

    mview.subject = inps.subject
    mview.comp = inps.comp
    mview.kind = inps.kind
    mview.tocm = inps.tocm

    mview.name = inps.name

    # Plotting tides.

    if inps.subject == 'secular':
        mview.show_secular()

    elif inps.subject in mview.tide_list:
        mview.show_tide()

    elif inps.subject[0:5] == 'resid':
        mview.show_resid()
    
    elif inps.subject[0:8] == 'speedmap':
        mview.show_speedmap()

    elif inps.subject[0:8] == 'other_1':
        mview.show_other_1()

    elif inps.subject[0:9] == 'temporary':
        mview.show_temporary()

    elif inps.subject[0:8] == 'dem':
        mview.show_dem()

    elif inps.subject[0:8] == 'basemap':
        mview.show_basemap()

    return

if __name__ == '__main__':
    main()
