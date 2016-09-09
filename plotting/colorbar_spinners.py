import matplotlib
matplotlib.use('qt4agg')


import matplotlib.pyplot as plt
import numpy as np
import threading
from time import sleep


class Spinner:
    def __init__(self, axes, initial_value=0, call_back=None, call_back_frequency=10):
        self._mouse_over = False
        self._is_focused = False
        self._canvas = axes.get_figure().canvas
        self._axes = axes
        self._axes.get_xaxis().set_visible(False)
        self._axes.get_yaxis().set_visible(False)
        self.text = None
        self._keyboard_cid = None
        self._value = None
        self._delta = 5
        self.set_value(initial_value)
        self._canvas.mpl_connect('axes_enter_event', self._enter_axis)
        self._canvas.mpl_connect('axes_leave_event', self._exit_axis)
        self._canvas.mpl_connect('button_press_event', self._clicked)
        self._canvas.mpl_connect('key_release_event', self._key_released)
        self._is_pressed = {'up': False, 'down': False}
        self._value_updater = threading.Thread(target=self._update_continuous)
        self._value_updater.start()
        self._call_back = call_back
        self._changed = False
        if call_back_frequency:
            self._call_back_delay = 1.0/call_back_frequency
        else:
            self._call_back_delay = None
        self._call_back_worker = threading.Thread(target=self._call_back_caller)
        self._call_back_worker.start()

    def _call_back_caller(self):
        if self._call_back_delay and self._call_back:
            while 1:
                sleep(self._call_back_delay)
                if self._changed:
                    self._changed = False
                    try:
                        self._call_back(self._value)
                    except Exception as e:
                        raise e

    def set_value(self, value):
        self._changed = True
        self._value = value
        if self.text is not None:
            self.text.remove()
        self.text = self._axes.text(0, .5, '%.3f' % self._value)
        self._canvas.draw()

    def _enter_axis(self, event):
        if event.inaxes == self._axes:
            self._mouse_over = True

    def _exit_axis(self, event):
        if event.inaxes == self._axes:
            self._mouse_over = False

    def _clicked(self, event):
        if self._mouse_over:
            self._get_focus()
        else:
            if self._is_focused:
                self._release_focus()

    def _get_focus(self):
        self._is_focused = True
        self._keyboard_cid = self._canvas.mpl_connect('key_press_event', self._key_pressed)

    def _release_focus(self):
        self._is_focused = False
        self._canvas.mpl_disconnect(self._keyboard_cid)

    def _key_pressed(self, event):
        print repr(event.key)
        if event.key == 'up':
            self._is_pressed['up'] = True
        elif event.key == 'down':
            self._is_pressed['down'] = True
        if event.key == 'enter':
            self._release_focus()

    def _key_released(self, event):
        if event.key == 'up':
            self._is_pressed['up'] = False
        elif event.key == 'down':
            self._is_pressed['down'] = False

    def _increment(self):
        self.set_value(self._value + self._delta)

    def _decrement(self):
        self.set_value(self._value - self._delta)

    def _update_continuous(self):
        while 1:
            sleep(.02)
            if self._is_pressed['up']:
                self._increment()
            if self._is_pressed['down']:
                self._decrement()


image = np.arange(0, 100).reshape(10,10)

plotaxes = plt.subplot2grid((15,15), (0,0), colspan=14, rowspan=15 )

colorbaraxes = plt.subplot2grid((21,15), (2,14), rowspan=17)
control1axes = plt.subplot2grid((21,15), (20,14))
control2axes = plt.subplot2grid((21,15), (0,14))

# control2axes.axis('off')
#control1axes.text(0, .5, 'Spinner1',bbox=dict(facecolor='white', alpha=0.5))
# control2axes.text(0, .5, 'Spinner2',bbox=dict(facecolor='white', alpha=0.5))
im = plotaxes.imshow(image)
plt.colorbar(mappable=im, cax=colorbaraxes)

clbck2 = lambda x: (im.set_clim(None, x))
clbck1 = lambda x: (im.set_clim(x,None))
spinner1 = Spinner(axes=control1axes, call_back=clbck1,call_back_frequency=100)
spinner2 = Spinner(axes=control2axes,initial_value=100, call_back=clbck2, call_back_frequency=100)
plt.show()
