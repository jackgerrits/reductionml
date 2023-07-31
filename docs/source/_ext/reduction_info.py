from docutils.parsers.rst import Directive
import reductionml_docs_extension
import docutils

class ReductionConfig(Directive):

    required_arguments = 1

    def run(self):
        parser = docutils.parsers.rst.Parser()
        document = self.state.document.copy()

        reduction_name = self.arguments[0]
        content = f"""
Typename: ``{reduction_name}``

.. csv-table::
    :header: "Property", "Type", "Default value"
    :align: left

"""

        info = reductionml_docs_extension.get_reduction_info(reduction_name)
        for prop in info["properties"]:
            name = prop['name']
            proptype = prop['type']

            if isinstance(prop["default"], dict) and "typename" in prop["default"]:
                proptype = "reduction"
                default_value = f"``{{\"typename\": \"{prop['default']['typename']}\"}}`` - :ref:`{prop['default']['typename']}`"
            else:
                default_value = f"``{str(prop['default'])}``"
            default_value = default_value.replace("\"", "\"\"")
            content += f"""    "``{name}``", "{proptype}", "{default_value}"\n """
        content += "\n\n"

        parser.parse(content, document)
        return document.children


def setup(app):
    app.add_directive("reduction_config", ReductionConfig)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
