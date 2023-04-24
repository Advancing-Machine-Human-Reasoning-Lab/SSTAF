import sys
import json
import predict_word

ALLOWED_COMMANDS = ["predict", "setTZero", "advanceClock"]
STOP_COMMANDS = ["break", "quit"]


def decompose_command(s):
    seq_num, command, args = s.split(maxsplit=2) + [""] * (2 - s.count(" "))

    try:
        seq_num = int(seq_num)
    except ValueError:
        seq_num = f"incompatible seq_num: {seq_num}"

    if command not in ALLOWED_COMMANDS:
        command = f"unknown command: {command}"

    try:
        args = args.replace("\'", "\"")
        args_check = json.loads(args)  # check if arg is in json
        args = json.dumps(args_check)  # convert back to JSON string
    except json.JSONDecodeError:
        args = f"poorly formed JSON: {args}"

    return seq_num, command, args


def send_result(msg):
    output = f"{msg}"
    #
    # Strip any embedded \n's from the output so it is all on one
    # space-delineated line.
    #
    clean = output.replace("\n", " ")
    sys.stdout.write(clean + '\n')
    sys.stdout.flush()


def send_error(seq_num, command, args, msg):
    send_result(f"{seq_num}\t {command}\t {args}\t error:{msg}")


t_zero = 0


def set_t_zero(args):
    global t_zero
    t_zero = int(args[0])
    return t_zero


def compute_capability(args):
    t_now = int(args[0])
    delta_t = t_now - t_zero
    max_t = 18 * 3600 * 1000

    capability = 1.0
    if t_now <= t_zero:
        capability = 1.0
    else:
        capability = 1 - (float(delta_t) / float(max_t))
    return capability


def main():
    #
    # Enter the main command processing loop.
    #
    usr_command = sys.stdin.readline().strip()

    while usr_command not in STOP_COMMANDS:
        seq_num, command, args = decompose_command(usr_command)
        #
        # Dispatch to appropriate method
        # I'm sure there is a more Pythonic way.
        #
        try:
            if command == ALLOWED_COMMANDS[0]:
                result = predict_word.main(args)
                result = result.replace("\"", "\'")
            elif command == ALLOWED_COMMANDS[1]:
                t0 = set_t_zero([args])
                result = str(t0)
            elif command == ALLOWED_COMMANDS[2]:
                capability = compute_capability([args])
                result = str(capability)
            else:
                raise Exception
            send_result(f"{seq_num} ok {command} {result}")
        except Exception as err:
            send_error(seq_num, command, args, err)
        #
        # Read the next line.
        #
        usr_command = sys.stdin.readline().strip()


if __name__ == "__main__":
    main()
