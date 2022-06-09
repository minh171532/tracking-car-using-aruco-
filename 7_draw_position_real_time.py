import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
filename = 'main_data.txt'
# creating a subplot
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    file = open(filename,'r')
    data = file.read()
    lines = data.split('\n')
    lines.remove('')
    xs = []
    ys = []
    for line in lines:
        x,y = line.split(',')[2:]
        xs.append(float(x))
        ys.append(float(y))
    xs = xs[-300:]
    ys = ys[-300:]

    ax1.clear()
    ax1.scatter(xs,ys)
    plt.xlim([0,500])
    plt.ylim([0,500])
    plt.xlabel('X coordinate persion(cm)')
    plt.ylabel('Y coordinate persion (cm)')
    plt.title('position')
    file.close

ani = animation.FuncAnimation(fig, animate, interval=200)
plt.show()