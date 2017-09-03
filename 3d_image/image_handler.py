import numpy as np

class ImageHandler(object):
    COLORMAP = 'gray'

    def __init__(self, path):
        self.path = path
        self.header = self.read_header()
        self.data = self.read_data()
        self.zone_slice_list = self.get_zone_slice_list()
        self.zone_crop_list = self.get_zone_crop_list()

    @staticmethod
    def get_zone_slice_list():
        # Divide the available space on an image into 16 sectors. In the [0] image these
        # zones correspond to the TSA threat zones.  But on rotated images, the slice
        # list uses the sector that best shows the threat zone
        sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
        sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
        sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
        sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)

        # sector 5 is used for both threat zone 5 and 17
        sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32)

        sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
        sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
        sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
        sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
        sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
        sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
        sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
        sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
        sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
        sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
        sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

        # Each element in the zone_slice_list contains the sector to use in the call to roi()
        zone_slice_list = [ [ # threat zone 1
                              sector01_pts, sector01_pts, sector01_pts, None,
                              None, None, sector03_pts, sector03_pts,
                              sector03_pts, sector03_pts, sector03_pts,
                              None, None, sector01_pts, sector01_pts, sector01_pts ],

                            [ # threat zone 2
                              sector02_pts, sector02_pts, sector02_pts, None,
                              None, None, sector04_pts, sector04_pts,
                              sector04_pts, sector04_pts, sector04_pts, None,
                              None, sector02_pts, sector02_pts, sector02_pts ],

                            [ # threat zone 3
                              sector03_pts, sector03_pts, sector03_pts, sector03_pts,
                              None, None, sector01_pts, sector01_pts,
                              sector01_pts, sector01_pts, sector01_pts, sector01_pts,
                              None, None, sector03_pts, sector03_pts ],

                            [ # threat zone 4
                              sector04_pts, sector04_pts, sector04_pts, sector04_pts,
                              None, None, sector02_pts, sector02_pts,
                              sector02_pts, sector02_pts, sector02_pts, sector02_pts,
                              None, None, sector04_pts, sector04_pts ],

                            [ # threat zone 5
                              sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                              sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                              None, None, None, None,
                              None, None, None, None ],

                            [ # threat zone 6
                              sector06_pts, None, None, None,
                              None, None, None, None,
                              sector07_pts, sector07_pts, sector06_pts, sector06_pts,
                              sector06_pts, sector06_pts, sector06_pts, sector06_pts ],

                            [ # threat zone 7
                              sector07_pts, sector07_pts, sector07_pts, sector07_pts,
                              sector07_pts, sector07_pts, sector07_pts, sector07_pts,
                              None, None, None, None,
                              None, None, None, None ],

                            [ # threat zone 8
                              sector08_pts, sector08_pts, None, None,
                              None, None, None, sector10_pts,
                              sector10_pts, sector10_pts, sector10_pts, sector10_pts,
                              sector08_pts, sector08_pts, sector08_pts, sector08_pts ],

                            [ # threat zone 9
                              sector09_pts, sector09_pts, sector08_pts, sector08_pts,
                              sector08_pts, None, None, None,
                              sector09_pts, sector09_pts, None, None,
                              None, None, sector10_pts, sector09_pts ],

                            [ # threat zone 10
                              sector10_pts, sector10_pts, sector10_pts, sector10_pts,
                              sector10_pts, sector08_pts, sector10_pts, None,
                              None, None, None, None,
                              None, None, None, sector10_pts ],

                            [ # threat zone 11
                              sector11_pts, sector11_pts, sector11_pts, sector11_pts,
                              None, None, sector12_pts, sector12_pts,
                              sector12_pts, sector12_pts, sector12_pts, None,
                              sector11_pts, sector11_pts, sector11_pts, sector11_pts ],

                            [ # threat zone 12
                              sector12_pts, sector12_pts, sector12_pts, sector12_pts,
                              sector12_pts, sector11_pts, sector11_pts, sector11_pts,
                              sector11_pts, sector11_pts, sector11_pts, None,
                              None, sector12_pts, sector12_pts, sector12_pts ],

                            [ # threat zone 13
                              sector13_pts, sector13_pts, sector13_pts, sector13_pts,
                              None, None, sector14_pts, sector14_pts,
                              sector14_pts, sector14_pts, sector14_pts, None,
                              sector13_pts, sector13_pts, sector13_pts, sector13_pts ],

                            [ # sector 14
                              sector14_pts, sector14_pts, sector14_pts, sector14_pts,
                              sector14_pts, None, sector13_pts, sector13_pts,
                              sector13_pts, sector13_pts, sector13_pts, None,
                              None, None, None, None ],

                            [ # threat zone 15
                              sector15_pts, sector15_pts, sector15_pts, sector15_pts,
                              None, None, sector16_pts, sector16_pts,
                              sector16_pts, sector16_pts, None, sector15_pts,
                              sector15_pts, None, sector15_pts, sector15_pts ],

                            [ # threat zone 16
                              sector16_pts, sector16_pts, sector16_pts, sector16_pts,
                              sector16_pts, sector16_pts, sector15_pts, sector15_pts,
                              sector15_pts, sector15_pts, sector15_pts, None,
                              None, None, sector16_pts, sector16_pts ],

                            [ # threat zone 17
                              None, None, None, None,
                              None, None, None, None,
                              sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                              sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]

        return zone_slice_list

    @staticmethod
    def get_zone_crop_list():
        # crop dimensions, upper left x, y, width, height
        sector_crop_list = [[ 50,  50, 250, 250], # sector 1
                            [  0,   0, 250, 250], # sector 2
                            [ 50, 250, 250, 250], # sector 3
                            [250,   0, 250, 250], # sector 4
                            [150, 150, 250, 250], # sector 5/17
                            [200, 100, 250, 250], # sector 6
                            [200, 150, 250, 250], # sector 7
                            [250,  50, 250, 250], # sector 8
                            [250, 150, 250, 250], # sector 9
                            [300, 200, 250, 250], # sector 10
                            [400, 100, 250, 250], # sector 11
                            [350, 200, 250, 250], # sector 12
                            [410,   0, 250, 250], # sector 13
                            [410, 200, 250, 250], # sector 14
                            [410,   0, 250, 250], # sector 15
                            [410, 200, 250, 250], # sector 16
                           ]

        # Each element in the zone_slice_list contains the sector to use in the call to roi()
        zone_crop_list =  [ [ # threat zone 1
                              sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None,
                              None, None, sector_crop_list[2], sector_crop_list[2],
                              sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], None,
                              None, sector_crop_list[0], sector_crop_list[0],
                              sector_crop_list[0] ],

                            [ # threat zone 2
                              sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None,
                              None, None, sector_crop_list[3], sector_crop_list[3],
                              sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
                              None, None, sector_crop_list[1], sector_crop_list[1],
                              sector_crop_list[1] ],

                            [ # threat zone 3
                              sector_crop_list[2], sector_crop_list[2], sector_crop_list[2],
                              sector_crop_list[2], None, None, sector_crop_list[0],
                              sector_crop_list[0], sector_crop_list[0], sector_crop_list[0],
                              sector_crop_list[0], sector_crop_list[0], None, None,
                              sector_crop_list[2], sector_crop_list[2] ],

                            [ # threat zone 4
                              sector_crop_list[3], sector_crop_list[3], sector_crop_list[3],
                              sector_crop_list[3], None, None, sector_crop_list[1],
                              sector_crop_list[1], sector_crop_list[1], sector_crop_list[1],
                              sector_crop_list[1], sector_crop_list[1], None, None,
                              sector_crop_list[3], sector_crop_list[3] ],

                            [ # threat zone 5
                              sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                              sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                              sector_crop_list[4], sector_crop_list[4],
                              None, None, None, None, None, None, None, None ],

                            [ # threat zone 6
                              sector_crop_list[5], None, None, None, None, None, None, None,
                              sector_crop_list[6], sector_crop_list[6], sector_crop_list[5],
                              sector_crop_list[5], sector_crop_list[5], sector_crop_list[5],
                              sector_crop_list[5], sector_crop_list[5] ],

                            [ # threat zone 7
                              sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                              sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                              sector_crop_list[6], sector_crop_list[6],
                              None, None, None, None, None, None, None, None ],

                            [ # threat zone 8
                              sector_crop_list[7], sector_crop_list[7], None, None, None,
                              None, None, sector_crop_list[9], sector_crop_list[9],
                              sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                              sector_crop_list[7], sector_crop_list[7], sector_crop_list[7],
                              sector_crop_list[7] ],

                            [ # threat zone 9
                              sector_crop_list[8], sector_crop_list[8], sector_crop_list[7],
                              sector_crop_list[7], sector_crop_list[7], None, None, None,
                              sector_crop_list[8], sector_crop_list[8], None, None, None,
                              None, sector_crop_list[9], sector_crop_list[8] ],

                            [ # threat zone 10
                              sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                              sector_crop_list[9], sector_crop_list[9], sector_crop_list[7],
                              sector_crop_list[9], None, None, None, None, None, None, None,
                              None, sector_crop_list[9] ],

                            [ # threat zone 11
                              sector_crop_list[10], sector_crop_list[10], sector_crop_list[10],
                              sector_crop_list[10], None, None, sector_crop_list[11],
                              sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                              sector_crop_list[11], None, sector_crop_list[10],
                              sector_crop_list[10], sector_crop_list[10], sector_crop_list[10] ],

                            [ # threat zone 12
                              sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                              sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                              sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                              sector_crop_list[11], sector_crop_list[11], None, None,
                              sector_crop_list[11], sector_crop_list[11], sector_crop_list[11] ],

                            [ # threat zone 13
                              sector_crop_list[12], sector_crop_list[12], sector_crop_list[12],
                              sector_crop_list[12], None, None, sector_crop_list[13],
                              sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                              sector_crop_list[13], None, sector_crop_list[12],
                              sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],

                            [ # sector 14
                              sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                              sector_crop_list[13], sector_crop_list[13], None,
                              sector_crop_list[13], sector_crop_list[13], sector_crop_list[12],
                              sector_crop_list[12], sector_crop_list[12], None, None, None,
                              None, None ],

                            [ # threat zone 15
                              sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
                              sector_crop_list[14], None, None, sector_crop_list[15],
                              sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                              None, sector_crop_list[14], sector_crop_list[14], None,
                              sector_crop_list[14], sector_crop_list[14] ],

                            [ # threat zone 16
                              sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                              sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                              sector_crop_list[14], sector_crop_list[14], sector_crop_list[14],
                              sector_crop_list[14], sector_crop_list[14], None, None, None,
                              sector_crop_list[15], sector_crop_list[15] ],

                            [ # threat zone 17
                              None, None, None, None, None, None, None, None,
                              sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                              sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                              sector_crop_list[4], sector_crop_list[4] ] ]

        return zone_crop_list

    def read_header(self):
        h = {}
        with open(self.path, 'r+b') as fid:
            h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
            h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
            h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
            h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
            h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
            h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
            h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
            h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
            h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
            h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
            h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
            h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
            h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
            h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
        return h

    def read_data(self):
        pass
