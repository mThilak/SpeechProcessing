#==============================================================================
#Project Modules
#==============================================================================
import matplotlib.pyplot as pyplot
#==============================================================================

def plotY(signal,stem = False,title='',xLabel='',yLabel=''):
    try:
        signal = list(signal)
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        
        if not stem:
            plotted = pyplot.plot(signal)
        else:
            plotted = pyplot.stem(signal)
        
        pyplot.show()
        return plotted
    except:
        print('Error in Plot Signal')
        print('The function takes only List Arguements')
         
def plotXY(x=[],y=[],stem = False,title='',xLabel='',yLabel=''):
    try:
        pyplot.figure()
        pyplot.title(title)
        pyplot.xlabel(xLabel)
        pyplot.ylabel(yLabel)
        pyplot.xlim([x[0],x[-1]])
        
        if not stem:
            plotted = pyplot.plot(x,y)
        else:
            plotted = pyplot.stem(x,y)
            
        pyplot.show()
        return plotted
    
    except:
        print('Error in Plotting Signal')
        print('The size of lists must be same')