#!/usr/bin/env python3
"""
Универсальность и скейлинговые свойства перехода к хаосу
в обобщённом отображении параболы: x_{n+1} = 1 - μ·|x_n|^z

Запуск: python main.py
"""

from gui import ChaosApp


def main():
    app = ChaosApp()
    app.run()


if __name__ == "__main__":
    main()
