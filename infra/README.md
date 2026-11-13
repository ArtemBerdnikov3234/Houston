# Инфраструктура

`docker-compose.yml` запускает Qdrant 1.19.2, публикует REST-порт 6333 и gRPC-порт 6334. Постоянные данные контейнера сохраняются в исключённом из Git каталоге `qdrant_storage/` в корне проекта.

Запуск из корня: `docker compose -f infra/docker-compose.yml up -d`.

## Содержимое и назначение

В папке хранится конфигурация инфраструктуры, необходимой Python-модулям. Сейчас Compose содержит один сервис — Qdrant; Python-скрипты выполняются в локальном виртуальном окружении. Отдельного контейнера приложения здесь пока нет.

Файл фиксирует образ Qdrant, имя контейнера `aerorag-qdrant`, порты и каталог постоянного хранения. Путь `../qdrant_storage` отсчитывается от Compose-файла и указывает на каталог в корне проекта. Это позволяет сохранять индекс при пересоздании контейнера.

## Проверка и управление

После запуска Docker Engine выполните команду выше. Состояние сервиса: `docker compose -f infra/docker-compose.yml ps`; логи: `docker compose -f infra/docker-compose.yml logs qdrant`. Веб-панель доступна по адресу `http://localhost:6333/dashboard`, проверка доступности — `http://localhost:6333/healthz`.

Остановка: `docker compose -f infra/docker-compose.yml down`. Файлы в `qdrant_storage/` при этой команде сохраняются. Compose запускает сервер, но не наполняет коллекцию: для этого отдельно используется модуль `src.retrieval.index_qdrant`.
