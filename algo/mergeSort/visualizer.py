import matplotlib.pyplot as plt
import numpy as np
import time


def setup_highlight_bars(data):
    """
    Initializes the bar chart with a specified data array.
    
    :param data: List or array of data values to be plotted as bars.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    # Store bars for future reference
    setup_highlight_bars.bars = plt.bar(range(len(data)), data, color='blue')
    plt.title('Bar Chart with Highlighted Bars')
    plt.xlabel('Index')
    plt.ylabel('Value')

def highlightBars(highlight_indexes):
    """
    Updates the bar chart to highlight specific bars.
    
    :param highlight_indexes: List or array of indexes to be highlighted.
    """
    # Reset all bars to blue initially
    for bar in setup_highlight_bars.bars:
        bar.set_color('blue')
    
    # Highlight specified bars in red
    for i in highlight_indexes:
        setup_highlight_bars.bars[i].set_color('red')
    
    plt.draw()
    plt.pause(pt)


def highlightBarsNO(highlight_indexes):
    """
    Updates the bar chart to highlight specific bars.
    
    :param highlight_indexes: List or array of indexes to be highlighted.
    """
    
    # Highlight specified bars in red
    for i in highlight_indexes:
        setup_highlight_bars.bars[i].set_color('red')
    
    plt.draw()
    #### plt.pause(pt)

def updateData(new_data):
    """
    Updates the graph with different data of the same length as the original.
    
    :param new_data: List or array of new data values to update the bars' heights.
    """
    # Update the heights of the bars with new data
    for bar, new_height in zip(setup_highlight_bars.bars, new_data):
        bar.set_height(new_height)
    
    plt.draw()
    plt.pause(pt)

s = 0
c = 0

def update_title():
    
    plt.title("Switches: {sw}, comparisons: {co}".format(sw=s,co=c), color='white')  # Set the new title with white color for visibility
    plt.draw()


def switch(data,i1,i2):
    global s
    if i1 == i2: return data
    highlightBars([i1,i2])
    data[i1], data[i2] = data[i2], data[i1]
    s+=1
    update_title()
    updateData(data)
    highlightBars([i1,i2])
    return data

def isBigger(data,i1,i2):
    global c
    c+=1
    update_title()
    highlightBars([i1,i2])
    return data[i1]>data[i2]

def isSmaller(da,i1,i2):
    global c
    c+=1
    update_title()
    highlightBars([i1,i2])
    return data[i1]<data[i2]

def insertionSort(A):
    print(A)
    s=False
    
    si=1
    while not s:
        s=True
        for i in range(len(A)-si):
            if isBigger(A,i,i+1):
                data=switch(A,i,i+1)
                s=False
        si+=1
    highlightBars([])
    print("DONE!")

#################################################################################
    
def quickSort(array, low=None, high=None):
    if low is None or high is None:
        low = 0
        high = len(array) - 1

    if (low < high):
        pi = partition(array, low, high)

        array=quickSort(array, low, pi - 1)
        array=quickSort(array, pi + 1, high)
    return(array)
    
def partition(array, low, high):
    pivot = array[high]
    
    i = (low - 1)

    for j in range(low,high):
        if isSmaller(array,j, high):
            i+=1    
            array = switch(array,i,j)
            

    array = switch(array,i+1,high)
    return (i + 1)

#################################################################################

pt = 0.01

data = np.arange(1, 40)[::-1]
np.random.shuffle(data)

setup_highlight_bars(data)

quickSort(data)
highlightBars([])

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the window open until manually closed




