# importing libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import dateutil
import DateTime
import pandas as pd

fig = plt.figure()
filename = 'main_data.txt'
# creating a subplot
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    file = open(filename, 'r')
    data = file.read()
    lines = data.split('\n')
    lines.remove('')
    # lines = set(lines)
    # lines = list(lines)
    # xs = []
    ys = []
    for line in lines:
        datetime_value, y = line.split(',')[:2]  # Delimiter is comma
        # xs.append(dateutil.parser.parse(datetime_value))
        ys.append(float(y))
    # xs = xs[-10:]
    ys = ys[-100:]
    # ax = plt.gca()
    ## ax.set_xticks(xs)

    # xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    # ax.xaxis.set_major_formatter(xfmt)
    ax1.clear()
    ax1.plot(ys, ".-")
    plt.ylim([0,10])
    plt.xlabel('Date time')
    # plt.xticks(rotation=25)
    plt.ylabel('speed')
    plt.title('velocity real time')
    file.close


ani = animation.FuncAnimation(fig, animate, interval=200)
plt.show()


