import datetime
import os
import toml

config = toml.load(os.path.join(os.getcwd(), os.pardir, 'ai_fsi_config.toml'))
log_path = os.path.join(os.getcwd(), os.pardir,
                        *config['main_settings']['save_log_path'])  # -> 'ai_logs/'


def write_log(log_name, text_in: list, cam_ip=''):
    """
    Функция для записи логов
    Args:
        param log_name: log name
        param text_in: text to write list format
        cam_ip: str

    :return: None

    """

    log_year = str(datetime.datetime.now().strftime('%Y'))
    log_month = str(datetime.datetime.now().strftime('%m'))
    log_day = str(datetime.datetime.now().strftime('%d'))
    main_log_path = os.path.join(log_path, log_year, log_month, log_day, '')

    if len(cam_ip):
        main_log_path = os.path.join(log_path, log_year, log_month, log_day, cam_ip, '')

    if not os.path.exists(main_log_path):
        os.makedirs(main_log_path)

    text_in = list(map(str, text_in))
    text_in.insert(0, str(datetime.datetime.now().strftime('%d_%m_%Y___%H:%M:%S:%f')))
    end_string = " ".join(text_in)

    if not os.path.exists(main_log_path):
        with open(main_log_path + log_name + '.log', 'w') as file_config:  # create log file
            file_config.write(end_string + "\n")
            file_config.close()

    elif os.path.exists(main_log_path):
        with open(main_log_path + log_name + '.log', 'a') as file_config:  # open log file
            file_config.write(end_string + "\n")
            file_config.close()
    else:
        print("LOG path error")


def write_log_one_line(log_name, text_in: list):
    """
    Функция для записи логов в одну строку

    :param log_name: log name
    :param text_in: text to write

    :return: None
    """
    log_year = str(datetime.datetime.now().strftime('%Y'))
    log_month = str(datetime.datetime.now().strftime('%m'))
    log_day = str(datetime.datetime.now().strftime('%d'))
    main_log_path = os.path.join(log_path, log_year, log_month, log_day, '')
    if not os.path.exists(main_log_path):
        os.makedirs(main_log_path)

    text_in = list(map(str, text_in))
    text_in.insert(0, str(datetime.datetime.now().strftime('%d_%m_%Y___%H:%M:%S:%f')))
    end_string = " ".join(text_in)

    if not os.path.exists(main_log_path):
        with open(main_log_path + log_name + '.log', 'w') as file_config:  # create log file
            file_config.write(end_string + "\n")
            file_config.close()

    elif os.path.exists(main_log_path):
        with open(main_log_path + log_name + '.log', 'w') as file_config:  # open log file
            file_config.write(end_string + "\n")
            file_config.close()
    else:
        print("LOG path error")


if __name__ == '__main__':
    write_log('test_log', ['test_line'], '35')
    write_log('test_log2', ['test_line'])
    write_log('3_yolo+hik_alg_out', ["192.168.19.85", 'img', 'number_out_yolo', 'error_message_yolo',
                                     'max_proba_image_map_hik', 'number_out_hik', 'error_message_hik'])
