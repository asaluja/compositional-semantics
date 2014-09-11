#!/usr/bin/python -tt

'''
File: visualize_parameters.py
Date: July 21, 2014
Description: this script visualizes the non-zero values of the
combiner function parameters.  
'''

import sys, commands, string, cPickle, math
import numpy as np
import pylab as plt
import matplotlib as mpl

def main():
    parameter, intercept = cPickle.load(open(sys.argv[1], 'rb'))
    outFile = mpl.backends.backend_pdf.PdfPages(sys.argv[2])
    numCharts = parameter.shape[0]
    chartsPerRow = int(sys.argv[3])
    chartsPerCol = int(sys.argv[4])
    chartsPerCell = chartsPerRow * chartsPerCol
    num_subplots = int(math.ceil(float(numCharts) / chartsPerCell))
    for sp in xrange(num_subplots):
        chartNum = 0
        coordinate = sp*chartsPerCell
        f, axes_tuples = plt.subplots(chartsPerCol, chartsPerRow, sharey=True, sharex=True)
        while chartNum < chartsPerCell:
            chartX = chartNum / chartsPerRow #truncates to nearest integer
            chartY = chartNum % chartsPerRow
            ax1 = axes_tuples[chartX][chartY]
            if coordinate < numCharts:
                param = parameter[coordinate, :, :]
                param[param==0] = np.nan
                heatmap = np.ma.array(param, mask=np.isnan(param))
                cmap = plt.cm.get_cmap('RdBu')
                cmap.set_bad('w', 1.)
                ax1.pcolor(heatmap, cmap=cmap, alpha=0.8)
            else:
                param = np.zeros((numCharts, numCharts))
                cmap = plt.cm.get_cmap('RdBu')
                ax1.pcolor(param, cmap=cmap, alpha=0.8)
            ax1.set_title('Dimension %d'%(coordinate+1))
            ax1.set_xlabel('Columns')
            ax1.set_ylabel('Rows')
            ax1.set_ylim([0,numCharts])
            ax1.set_xlim([0,numCharts])
            chartNum += 1
            coordinate += 1
        plt.tight_layout()
        outFile.savefig()
    outFile.close()        

if __name__ == "__main__":
    main()
