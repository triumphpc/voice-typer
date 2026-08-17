/*
 * Лаунчер VoiceTyper.app.
 *
 * Интерпретатор Python встроен прямо в этот бинарник, а не запускается
 * через exec отдельным процессом. Это принципиально: macOS выдаёт права
 * (Мониторинг ввода, Универсальный доступ, Микрофон) тому исполняемому файлу,
 * который их запросил. Если сделать exec на .../bin/python3.13 - образ процесса
 * подменяется, права уезжают на «Python», а приложение остаётся ни с чем,
 * причём молча: галочка в настройках стоит, а перехват клавиш не работает.
 *
 * Пути (интерпретатор, скрипт, site-packages) подставляются на этапе сборки
 * через -D, см. build_app.sh.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void die(const char *message) {
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "osascript -e 'display alert \"VoiceTyper\" message \"%s\" as critical'",
             message);
    system(cmd);
    exit(1);
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    if (access(VT_SCRIPT, R_OK) != 0) {
        die("Не найден " VT_SCRIPT "\n\nПроект переехал? Пересобери приложение: ./build_app.sh");
    }

    /* Пакеты берём из venv проекта, сам интерпретатор - системный framework. */
    setenv("PYTHONPATH", VT_SITE_PACKAGES, 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;

    status = PyConfig_SetBytesString(&config, &config.program_name, VT_PROGRAM_NAME);
    if (PyStatus_Exception(status)) goto fail;

    status = PyConfig_SetBytesString(&config, &config.run_filename, VT_SCRIPT);
    if (PyStatus_Exception(status)) goto fail;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto fail;

    PyConfig_Clear(&config);
    return Py_RunMain();

fail:
    PyConfig_Clear(&config);
    if (PyStatus_IsExit(status)) return status.exitcode;
    die("Не удалось запустить встроенный Python");
    return 1;
}
