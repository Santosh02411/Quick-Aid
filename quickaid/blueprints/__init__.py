"""One Blueprint per area of the product. `register_blueprints` is the single place that lists them."""


def register_blueprints(app) -> None:
    from quickaid.blueprints import admin, auth, conversations, history, images, pages, privacy, symptoms, system
    for module in (pages, system, auth, symptoms, images, conversations, history, privacy, admin):
        app.register_blueprint(module.bp)
