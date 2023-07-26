from docutils import nodes
from docutils.parsers.rst import Directive
import reductionml_docs_extension

class ReductionConfig(Directive):

    required_arguments = 1

    def run(self):
        reduction_name = self.arguments[0]
        info = reductionml_docs_extension.get_reduction_info(reduction_name)
        bullet_list = nodes.bullet_list()
        for prop in info["properties"]:
            item = nodes.list_item()
            item += nodes.Text(f"{prop['name']}(")
            item += nodes.literal(text=prop["type"])
            item += nodes.Text(f"), default=")

            if isinstance(prop["default"], dict) and "typename" in prop["default"]:
                value = f"{{\"typename\": \"{prop['default']['typename']}\"}}"
                item += nodes.literal(text=value)
            else:
                item += nodes.literal(text=prop["default"])

            bullet_list += item

        return [bullet_list]


def setup(app):
    app.add_directive("reduction_config", ReductionConfig)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
