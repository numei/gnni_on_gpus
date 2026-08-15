#include "gnni/dataset_io.hpp"
#include "gnni/synthetic_graph.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <unistd.h>

namespace {

struct Options {
  std::string model;
  int vertices = 0;
  std::size_t features = 0;
  std::uint64_t seed = 0;
  bool self_loops = false;
  bool force = false;
  std::filesystem::path output;
  double probability = -1.0;
  double average_degree = -1.0;
  int edges_per_new_vertex = 0;
  int neighbors = 0;
  double rewire_probability = -1.0;
};

long long parse_integer(const std::string& text, const std::string& name,
                        long long minimum, long long maximum) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + " requires an integer");
  }
  if (consumed != text.size() || value < minimum || value > maximum) {
    throw std::invalid_argument(name + " is outside its allowed range");
  }
  return value;
}

std::uint64_t parse_seed(const std::string& text) {
  std::size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("--seed requires an unsigned integer");
  }
  if (consumed != text.size()) {
    throw std::invalid_argument("--seed requires an unsigned integer");
  }
  return static_cast<std::uint64_t>(value);
}

double parse_real(const std::string& text, const std::string& name) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(name + " requires a real number");
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " requires a finite real number");
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    throw std::invalid_argument("missing model: expected er, ba, or ws");
  }
  if (std::string(argv[1]) == "--help") {
    std::cout
        << "Usage:\n"
        << "  gnni_dataset_generator er --vertices N "
           "(--probability P|--average-degree D) --features F --seed S "
           "[--self-loops] --output FILE [--force]\n"
        << "  gnni_dataset_generator ba --vertices N "
           "--edges-per-new-vertex M --features F --seed S "
           "[--self-loops] --output FILE [--force]\n"
        << "  gnni_dataset_generator ws --vertices N --neighbors K "
           "--rewire-probability B --features F --seed S "
           "[--self-loops] --output FILE [--force]\n";
    std::exit(0);
  }

  Options options;
  options.model = argv[1];
  if (options.model != "er" && options.model != "ba" &&
      options.model != "ws") {
    throw std::invalid_argument("unknown model: " + options.model);
  }

  std::unordered_map<std::string, std::string> values;
  std::unordered_set<std::string> flags;
  for (int index = 2; index < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--self-loops" || name == "--force") {
      if (!flags.insert(name).second) {
        throw std::invalid_argument("duplicate option: " + name);
      }
      continue;
    }
    if (name.rfind("--", 0) != 0 || index + 1 >= argc) {
      throw std::invalid_argument("option requires a value: " + name);
    }
    if (!values.emplace(name, argv[++index]).second) {
      throw std::invalid_argument("duplicate option: " + name);
    }
  }

  const auto required = [&](const std::string& name) -> const std::string& {
    const auto found = values.find(name);
    if (found == values.end()) {
      throw std::invalid_argument("missing required option: " + name);
    }
    return found->second;
  };
  options.vertices = static_cast<int>(parse_integer(
      required("--vertices"), "--vertices", 1,
      std::numeric_limits<int>::max()));
  options.features = static_cast<std::size_t>(parse_integer(
      required("--features"), "--features", 1,
      std::numeric_limits<int>::max()));
  options.seed = parse_seed(required("--seed"));
  options.output = required("--output");
  options.self_loops = flags.count("--self-loops") != 0;
  options.force = flags.count("--force") != 0;

  std::unordered_set<std::string> allowed = {
      "--vertices", "--features", "--seed", "--output"};
  if (options.model == "er") {
    const bool has_probability = values.count("--probability") != 0;
    const bool has_average = values.count("--average-degree") != 0;
    if (has_probability == has_average) {
      throw std::invalid_argument(
          "ER requires exactly one of --probability and --average-degree");
    }
    if (has_probability) {
      options.probability = parse_real(values.at("--probability"),
                                       "--probability");
      allowed.insert("--probability");
    } else {
      options.average_degree = parse_real(values.at("--average-degree"),
                                          "--average-degree");
      allowed.insert("--average-degree");
    }
  } else if (options.model == "ba") {
    options.edges_per_new_vertex = static_cast<int>(parse_integer(
        required("--edges-per-new-vertex"), "--edges-per-new-vertex", 1,
        std::numeric_limits<int>::max()));
    allowed.insert("--edges-per-new-vertex");
  } else {
    options.neighbors = static_cast<int>(parse_integer(
        required("--neighbors"), "--neighbors", 1,
        std::numeric_limits<int>::max()));
    options.rewire_probability = parse_real(
        required("--rewire-probability"), "--rewire-probability");
    allowed.insert("--neighbors");
    allowed.insert("--rewire-probability");
  }
  for (const auto& [name, value] : values) {
    static_cast<void>(value);
    if (allowed.count(name) == 0) {
      throw std::invalid_argument("option is not valid for " + options.model +
                                  ": " + name);
    }
  }
  return options;
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (char raw_character : value) {
    const auto character = static_cast<unsigned char>(raw_character);
    if (character == '"' || character == '\\') {
      escaped.push_back('\\');
      escaped.push_back(static_cast<char>(character));
    } else if (character == '\n') {
      escaped += "\\n";
    } else if (character < 0x20U) {
      throw std::invalid_argument("output path contains a control character");
    } else {
      escaped.push_back(static_cast<char>(character));
    }
  }
  return escaped;
}

void write_sidecar(const Options& options, std::uint64_t logical_edges,
                   std::size_t csr_entries,
                   std::uintmax_t output_bytes) {
  const std::filesystem::path sidecar = options.output.string() + ".json";
  if (!options.force &&
      std::filesystem::symlink_status(sidecar).type() !=
          std::filesystem::file_type::not_found) {
    throw std::runtime_error("sidecar already exists: " + sidecar.string());
  }
  const std::filesystem::path temporary =
      sidecar.string() + ".tmp-" + std::to_string(::getpid());
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("cannot create sidecar: " + temporary.string());
  }
  stream << "{\n"
         << "  \"format_version\": 1,\n"
         << "  \"generator\": \"" << options.model << "\",\n"
         << "  \"output\": \"" << json_escape(options.output.string())
         << "\",\n"
         << "  \"vertices\": " << options.vertices << ",\n"
         << "  \"feature_dim\": " << options.features << ",\n"
         << "  \"seed\": " << options.seed << ",\n"
         << "  \"logical_edges\": " << logical_edges << ",\n"
         << "  \"csr_entries\": " << csr_entries << ",\n"
         << "  \"directed\": false,\n"
         << "  \"self_loops\": "
         << (options.self_loops ? "true" : "false") << ",\n"
         << "  \"output_bytes\": " << output_bytes << ",\n";
  if (options.model == "er") {
    const double probability = options.probability >= 0.0
                                   ? options.probability
                                   : gnni::er_probability_from_average_degree(
                                         options.vertices,
                                         options.average_degree);
    stream << "  \"probability\": " << probability << "\n";
  } else if (options.model == "ba") {
    stream << "  \"edges_per_new_vertex\": "
           << options.edges_per_new_vertex << "\n";
  } else {
    stream << "  \"neighbors\": " << options.neighbors << ",\n"
           << "  \"rewire_probability\": "
           << options.rewire_probability << "\n";
  }
  stream << "}\n";
  stream.flush();
  if (!stream) {
    stream.close();
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed writing sidecar");
  }
  stream.close();
  std::error_code error;
  std::filesystem::rename(temporary, sidecar, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("failed installing sidecar: " + error.message());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::filesystem::path sidecar =
        options.output.string() + ".json";
    if (!options.force &&
        std::filesystem::symlink_status(sidecar).type() !=
            std::filesystem::file_type::not_found) {
      throw std::runtime_error("sidecar already exists: " + sidecar.string());
    }

    gnni::GeneratedGraph generated;
    if (options.model == "er") {
      const double probability = options.probability >= 0.0
                                   ? options.probability
                                   : gnni::er_probability_from_average_degree(
                                         options.vertices,
                                         options.average_degree);
      generated = gnni::generate_erdos_renyi(
          options.vertices, probability, options.seed, options.self_loops);
    } else if (options.model == "ba") {
      generated = gnni::generate_barabasi_albert(
          options.vertices, options.edges_per_new_vertex, options.seed,
          options.self_loops);
    } else {
      generated = gnni::generate_watts_strogatz(
          options.vertices, options.neighbors, options.rewire_probability,
          options.seed, options.self_loops);
    }

    const std::uint64_t logical_edges = generated.logical_edges;
    const std::size_t csr_entries = generated.graph.column_indices.size();
    const gnni::Dataset dataset = gnni::make_synthetic_dataset(
        std::move(generated), options.features,
        options.seed ^ UINT64_C(0xd1b54a32d192ed03));
    gnni::save_dataset(options.output, dataset, options.force);
    write_sidecar(options, logical_edges, csr_entries,
                  std::filesystem::file_size(options.output));
    std::cout << "dataset=" << options.output.string() << '\n'
              << "model=" << options.model << '\n'
              << "vertices=" << options.vertices << '\n'
              << "logical_edges=" << logical_edges << '\n'
              << "csr_entries=" << csr_entries << '\n'
              << "feature_dim=" << options.features << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dataset generation failed: " << error.what() << '\n';
    return 1;
  }
}
