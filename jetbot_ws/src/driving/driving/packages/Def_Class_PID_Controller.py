def calc_error_value(desired_output, measured_output):
    return desired_output - measured_output


class PID_Controller:
    """
    Class providing structure of a PID controller
    """

    def __init__(self, k_p, k_i, k_d, time_step=1. / 60.):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.e_last = 0
        self.e_sum = 0

        self.time_step = time_step

    def calc_controller_output(self, error_value):
        p = self.k_p * error_value

        self.e_sum += (error_value + self.e_last)
        i = self.k_i * (self.time_step / 2) * self.e_sum

        d = self.k_d / self.time_step * (error_value - self.e_last)

        self.e_last = error_value

        return p + i + d
